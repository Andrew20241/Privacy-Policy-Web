from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import networkx as nx
import requests
from bs4 import BeautifulSoup, Tag

try:
    from pyvis.network import Network
except ImportError:
    Network = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

USER_AGENT = (
    "Mozilla/5.0 (compatible; GeorgiaTechPolicyCrawler/1.0; "
    "+https://github.com/)"
)

DEFAULT_CONTENT_SELECTORS = [
    "main article",
    "main .node__content",
    "main .layout-content",
    "main .region-content",
    "main .field--name-body",
    "main .content",
    "article",
]

DEFAULT_EXCLUDE_SELECTORS = [
    "nav",
    "aside",
    "header",
    "footer",
    "form",
    "script",
    "style",
    "noscript",
    ".breadcrumb",
    ".pager",
    ".tabs",
    ".sidebar",
    ".region-sidebar-first",
    ".region-sidebar-second",
    ".block-menu",
    ".menu",
    ".contextual-links",
    ".print__link",
    ".social-media-links",
    ".skip-link",
]

CONTENT_HINTS = [
    "Reason for Policy",
    "Policy Statement",
    "Scope",
    "Definitions",
    "Procedures",
    "Responsibilities",
    "Enforcement",
    "Related Information",
    "Frequently Asked Questions",
]

HEAD_FALLBACK_STATUS_CODES = {403, 405, 429, 500, 501, 502, 503}
START_NODE_COLOR = "#f4b400"
INTERNAL_NODE_COLOR = "#2e86ab"
EXTERNAL_NODE_COLOR = "#a0a4a8"
DEAD_NODE_COLOR = "#d1495b"

@dataclass
class CrawlConfig:
    start_url: str
    output_dir: Path
    max_depth: int
    max_pages: int
    timeout: float
    delay: float
    allowed_domains: tuple[str, ...]
    content_selectors: tuple[str, ...]
    exclude_selectors: tuple[str, ...]
    follow_external: bool
    verify_ssl: bool

@dataclass
class StatusResult:
    requested_url: str
    final_url: str
    status_code: int | None
    error: str | None
    content_type: str | None = None

    @property
    def dead(self) -> bool:
        return self.error is not None or (self.status_code is not None and self.status_code >= 400)

@dataclass
class PageResult(StatusResult):
    title: str = ""
    site_name: str = ""
    html: str = ""

    @property
    def is_html(self) -> bool:
        return bool(self.html) and "html" in (self.content_type or "").lower()

@dataclass
class ExtractedLink:
    url: str
    text: str

class PolicyCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.graph = nx.DiGraph()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.status_cache: dict[str, StatusResult] = {}
        self.page_cache: dict[str, PageResult] = {}
        self.visited_pages: set[str] = set()

    def crawl(self) -> dict[str, object]:
        start_status = self.check_url(self.config.start_url)
        start_url = start_status.final_url
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        queued_urls = {start_url}

        while queue and len(self.visited_pages) < self.config.max_pages:
            url, depth = queue.popleft()
            queued_urls.discard(url)
            if url in self.visited_pages:
                continue

            page = self.fetch_page(url)
            canonical_url = page.final_url
            self.visited_pages.add(canonical_url)
            self.upsert_node(
                canonical_url,
                label=page.title or guess_label(canonical_url),
                title=page.title,
                status_code=page.status_code,
                error=page.error,
                dead=page.dead,
                internal=self.is_internal(canonical_url),
                visited=True,
                depth=depth,
                content_type=page.content_type or "",
                start=depth == 0,
            )

            if page.dead or not page.is_html:
                time.sleep(self.config.delay)
                continue

            soup = BeautifulSoup(page.html, "html.parser")
            content_root = self.select_content_root(soup)
            self.prune_non_content(content_root)

            for link in self.extract_links(content_root, canonical_url):
                status = self.check_url(link.url)
                target_url = status.final_url
                if target_url == canonical_url:
                    continue

                self.upsert_node(
                    target_url,
                    label=self.resolve_link_label(target_url, link.text, status),
                    title=self.resolve_link_title(target_url, link.text, status),
                    status_code=status.status_code,
                    error=status.error,
                    dead=status.dead,
                    internal=self.is_internal(target_url),
                    visited=target_url in self.visited_pages,
                    content_type=status.content_type or "",
                )
                self.add_edge(canonical_url, target_url, link.text)

                if self.should_follow(target_url, depth + 1):
                    if target_url not in self.visited_pages and target_url not in queued_urls:
                        queue.append((target_url, depth + 1))
                        queued_urls.add(target_url)

            time.sleep(self.config.delay)

        return self.export_outputs()

    def should_follow(self, url: str, next_depth: int) -> bool:
        if next_depth > self.config.max_depth:
            return False
        if self.config.follow_external:
            return urlsplit(url).scheme in {"http", "https"}
        return self.is_internal(url)

    def is_internal(self, url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        host = parsed.netloc.lower()
        return any(host == domain or host.endswith(f".{domain}") for domain in self.config.allowed_domains)

    def fetch_page(self, url: str) -> PageResult:
        cached = self.page_cache.get(url)
        if cached is not None:
            return cached

        try:
            response = self.session.get(
                url,
                allow_redirects=True,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            final_url = normalize_url(response.url)
            content_type = response.headers.get("Content-Type", "")
            title = ""
            html = ""
            if "html" in content_type.lower():
                html = response.text
                soup = BeautifulSoup(html, "html.parser")
                title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
                site_name = extract_site_name(soup, title)
            else:
                site_name = ""
            page = PageResult(
                requested_url=url,
                final_url=final_url,
                status_code=response.status_code,
                error=None if response.ok else f"HTTP {response.status_code}",
                content_type=content_type,
                title=title,
                site_name=site_name,
                html=html,
            )
        except requests.RequestException as exc:
            page = PageResult(
                requested_url=url,
                final_url=url,
                status_code=None,
                error=str(exc),
                content_type=None,
                title="",
                site_name="",
                html="",
            )

        self.page_cache[url] = page
        self.status_cache[url] = StatusResult(
            requested_url=page.requested_url,
            final_url=page.final_url,
            status_code=page.status_code,
            error=page.error,
            content_type=page.content_type,
        )
        return page

    def check_url(self, url: str) -> StatusResult:
        normalized = normalize_url(url)
        cached = self.status_cache.get(normalized)
        if cached is not None:
            return cached

        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"}:
            result = StatusResult(
                requested_url=normalized,
                final_url=normalized,
                status_code=None,
                error=f"Unsupported scheme: {parsed.scheme or 'unknown'}",
                content_type=None,
            )
            self.status_cache[normalized] = result
            return result

        response = None
        try:
            response = self.session.head(
                normalized,
                allow_redirects=True,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            if response.status_code in HEAD_FALLBACK_STATUS_CODES:
                response.close()
                response = self.session.get(
                    normalized,
                    allow_redirects=True,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                    stream=True,
                )

            result = StatusResult(
                requested_url=normalized,
                final_url=normalize_url(response.url),
                status_code=response.status_code,
                error=None if response.ok else f"HTTP {response.status_code}",
                content_type=response.headers.get("Content-Type", ""),
            )
        except requests.RequestException as exc:
            result = StatusResult(
                requested_url=normalized,
                final_url=normalized,
                status_code=None,
                error=str(exc),
                content_type=None,
            )
        finally:
            if response is not None:
                response.close()

        self.status_cache[normalized] = result
        return result

    def select_content_root(self, soup: BeautifulSoup) -> Tag:
        root = soup.select_one("main") or soup.select_one('[role="main"]') or soup.body or soup
        self.prune_global_noise(root)

        for selector in self.config.content_selectors:
            match = root.select_one(selector)
            if match is not None:
                return match

        candidates: list[Tag] = [root]
        for candidate in root.find_all(["article", "section", "div"], recursive=True):
            text = clean_text(candidate.get_text(" ", strip=True))
            if len(text) >= 250:
                candidates.append(candidate)

        best = max(candidates, key=self.score_candidate)
        return best

    def prune_global_noise(self, root: Tag) -> None:
        selector = ",".join(self.config.exclude_selectors)
        if selector:
            for node in list(root.select(selector)):
                node.decompose()

    def prune_non_content(self, root: Tag) -> None:
        self.prune_global_noise(root)
        for node in list(root.find_all(True)):
            if node is root:
                continue
            if getattr(node, "attrs", None) is None:
                continue
            if self.is_link_cluster(node):
                node.decompose()

    def is_link_cluster(self, node: Tag) -> bool:
        attrs = getattr(node, "attrs", None)
        if attrs is None:
            return False

        classes = " ".join(attrs.get("class", []))
        identifier = f"{attrs.get('id', '')} {classes}".lower()
        if any(token in identifier for token in ("sidebar", "nav", "menu", "breadcrumb", "pager")):
            return True

        links = len(node.find_all("a", href=True))
        list_items = len(node.find_all("li"))
        paragraphs = len(node.find_all("p"))
        headings = len(node.find_all(["h1", "h2", "h3", "h4"]))
        text_len = len(clean_text(node.get_text(" ", strip=True)))

        if links >= 8 and paragraphs <= 1 and headings <= 1:
            return True
        if list_items >= 6 and paragraphs == 0:
            return True
        if links >= 10 and text_len < 2500:
            return True
        return False

    def score_candidate(self, node: Tag) -> float:
        text = clean_text(node.get_text(" ", strip=True))
        text_len = len(text)
        paragraphs = len(node.find_all("p"))
        headings = len(node.find_all(["h1", "h2", "h3", "h4"]))
        links = len(node.find_all("a", href=True))
        list_items = len(node.find_all("li"))
        definition_items = len(node.find_all(["dt", "dd"]))
        table_rows = len(node.find_all("tr"))

        keyword_bonus = sum(15 for hint in CONTENT_HINTS if hint.lower() in text.lower())
        class_bonus = 0
        class_tokens = f"{node.get('id', '')} {' '.join(node.get('class', []))}".lower()
        if any(token in class_tokens for token in ("content", "body", "article", "main")):
            class_bonus += 10
        if any(token in class_tokens for token in ("sidebar", "nav", "menu")):
            class_bonus -= 30

        score = (
            (text_len / 120)
            + (paragraphs * 24)
            + (headings * 8)
            + (definition_items * 5)
            + (table_rows * 3)
            + keyword_bonus
            + class_bonus
        )
        score -= list_items * 6
        score -= links * 0.8
        if paragraphs == 0:
            score -= 35
        if links > paragraphs * 6 and paragraphs > 0:
            score -= (links - (paragraphs * 6)) * 2
        return score

    def extract_links(self, root: Tag, base_url: str) -> list[ExtractedLink]:
        found: dict[str, str] = {}
        for anchor in root.find_all("a", href=True):
            raw_href = anchor.get("href", "").strip()
            if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            resolved = normalize_url(urljoin(base_url, raw_href))
            if not resolved:
                continue

            text = clean_text(anchor.get_text(" ", strip=True)) or guess_label(resolved)
            if self.should_skip_link(resolved, text):
                continue
            found.setdefault(resolved, text)

        return [ExtractedLink(url=url, text=text) for url, text in found.items()]

    def should_skip_link(self, url: str, text: str) -> bool:
        cleaned_text = clean_text(text).lower()
        parsed = urlsplit(url)
        path = parsed.path.lower()

        if "printer-friendly" in cleaned_text or "printer friendly" in cleaned_text:
            return True
        if path.startswith("/book/export/") or "/book/export/" in path:
            return True
        return False

    def resolve_link_label(self, url: str, anchor_text: str, status: StatusResult) -> str:
        anchor_text = clean_text(anchor_text)
        if anchor_text and not is_generic_anchor_text(anchor_text):
            return guess_label(url, anchor_text)

        if not status.dead and "html" in (status.content_type or "").lower():
            page = self.fetch_page(url)
            if page.title:
                page_label = extract_page_label_from_title(page.title)
                if page_label and not is_generic_anchor_text(page_label):
                    return page_label[:120]
            if page.site_name:
                return page.site_name[:120]
            if page.title:
                return extract_site_name_from_title(page.title)[:120]

        return guess_label(url, anchor_text)

    def resolve_link_title(self, url: str, anchor_text: str, status: StatusResult) -> str:
        if not is_generic_anchor_text(anchor_text):
            return ""
        if not status.dead and "html" in (status.content_type or "").lower():
            page = self.fetch_page(url)
            return page.title
        return ""

    def upsert_node(self, url: str, **attrs: object) -> None:
        if url not in self.graph:
            self.graph.add_node(url)

        node = self.graph.nodes[url]
        for key, value in attrs.items():
            if value is None:
                continue
            if key == "label":
                if not node.get("label") or looks_generic(str(node.get("label"))):
                    node[key] = str(value)
            elif key == "title":
                if value:
                    node[key] = str(value)
                    if not node.get("label") or looks_generic(str(node.get("label"))):
                        node["label"] = str(value)
            elif key in {"dead", "internal", "visited", "start"}:
                node[key] = bool(value)
            elif key in {"depth", "status_code"}:
                node[key] = int(value) if value != "" else 0
            else:
                node[key] = str(value)

        node.setdefault("url", url)
        node.setdefault("label", guess_label(url))
        node.setdefault("title", "")
        node.setdefault("status_code", -1)
        node.setdefault("error", "")
        node.setdefault("dead", False)
        node.setdefault("internal", self.is_internal(url))
        node.setdefault("visited", False)
        node.setdefault("start", False)
        node.setdefault("depth", -1)
        node.setdefault("content_type", "")

    def add_edge(self, source: str, target: str, anchor_text: str) -> None:
        cleaned = clean_text(anchor_text)
        if self.graph.has_edge(source, target):
            existing = self.graph.edges[source, target].get("anchor_texts", "")
            self.graph.edges[source, target]["anchor_texts"] = merge_pipe_separated(existing, cleaned)
        else:
            self.graph.add_edge(source, target, anchor_texts=cleaned)

    def export_outputs(self) -> dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stem = slugify_url(self.config.start_url)

        graphml_path = self.config.output_dir / f"{stem}.graphml"
        pickle_path = self.config.output_dir / f"{stem}.networkx.pkl"
        nodes_csv_path = self.config.output_dir / f"{stem}.nodes.csv"
        edges_csv_path = self.config.output_dir / f"{stem}.edges.csv"
        summary_path = self.config.output_dir / f"{stem}.summary.json"
        html_path = self.config.output_dir / f"{stem}.html"
        png_path = self.config.output_dir / f"{stem}.networkx.png"

        nx.write_graphml(self.graph, graphml_path)
        self.write_networkx_pickle(pickle_path)
        self.write_nodes_csv(nodes_csv_path)
        self.write_edges_csv(edges_csv_path)

        summary = {
            "start_url": self.config.start_url,
            "pages_crawled": len(self.visited_pages),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "dead_nodes": sum(1 for _, data in self.graph.nodes(data=True) if data.get("dead")),
            "internal_nodes": sum(1 for _, data in self.graph.nodes(data=True) if data.get("internal")),
            "external_nodes": sum(1 for _, data in self.graph.nodes(data=True) if not data.get("internal")),
            "graphml": str(graphml_path),
            "networkx_pickle": str(pickle_path),
            "nodes_csv": str(nodes_csv_path),
            "edges_csv": str(edges_csv_path),
        }

        if Network is not None:
            self.write_html_graph(html_path)
            summary["html_graph"] = str(html_path)
        else:
            summary["html_graph"] = None
            summary["html_graph_note"] = "Install pyvis to generate the interactive HTML graph."

        if plt is not None:
            self.write_networkx_png(png_path)
            summary["networkx_png"] = str(png_path)
        else:
            summary["networkx_png"] = None
            summary["networkx_png_note"] = "Install matplotlib to generate the NetworkX PNG graph."

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        return summary

    def write_networkx_pickle(self, output_path: Path) -> None:
        with output_path.open("wb") as handle:
            pickle.dump(self.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def write_nodes_csv(self, output_path: Path) -> None:
        fieldnames = [
            "url",
            "label",
            "title",
            "status_code",
            "dead",
            "internal",
            "visited",
            "start",
            "depth",
            "error",
            "content_type",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for url, data in sorted(self.graph.nodes(data=True), key=lambda item: item[0]):
                writer.writerow(
                    {
                        "url": url,
                        "label": data.get("label", ""),
                        "title": data.get("title", ""),
                        "status_code": data.get("status_code", ""),
                        "dead": data.get("dead", False),
                        "internal": data.get("internal", False),
                        "visited": data.get("visited", False),
                        "start": data.get("start", False),
                        "depth": data.get("depth", ""),
                        "error": data.get("error", ""),
                        "content_type": data.get("content_type", ""),
                    }
                )

    def write_edges_csv(self, output_path: Path) -> None:
        fieldnames = ["source", "target", "anchor_texts", "target_dead"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for source, target, data in sorted(self.graph.edges(data=True), key=lambda item: (item[0], item[1])):
                writer.writerow(
                    {
                        "source": source,
                        "target": target,
                        "anchor_texts": data.get("anchor_texts", ""),
                        "target_dead": self.graph.nodes[target].get("dead", False),
                    }
                )

    def write_html_graph(self, output_path: Path) -> None:
        if Network is None:
            return

        network = Network(
            height="900px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#222222",
            directed=True,
            cdn_resources="remote",
        )
        network.set_options(
            """
            {
              "nodes": {
                "font": {"size": 14},
                "shape": "dot",
                "scaling": {"min": 12, "max": 36}
              },
              "edges": {
                "arrows": {"to": {"enabled": true}},
                "smooth": {"enabled": true, "type": "dynamic"}
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -3200,
                  "springLength": 180,
                  "springConstant": 0.03
                },
                "minVelocity": 0.75
              }
            }
            """
        )

        for url, data in self.graph.nodes(data=True):
            dead = bool(data.get("dead"))
            start = bool(data.get("start"))
            internal = bool(data.get("internal"))
            color = DEAD_NODE_COLOR if dead else START_NODE_COLOR if start else INTERNAL_NODE_COLOR if internal else EXTERNAL_NODE_COLOR
            title = (
                f"{data.get('label', url)}<br>"
                f"URL: {url}<br>"
                f"Status: {data.get('status_code', 'unknown')}<br>"
                f"Dead: {dead}<br>"
                f"Start node: {start}<br>"
                f"Visited: {bool(data.get('visited'))}<br>"
                f"Error: {data.get('error', '')}"
            )
            size = 28 if start else 24 if dead else 20 if internal else 16
            network.add_node(url, label=str(data.get("label", guess_label(url))), title=title, color=color, size=size)

        for source, target, data in self.graph.edges(data=True):
            edge_color = "#d1495b" if self.graph.nodes[target].get("dead") else "#9aa4ad"
            network.add_edge(
                source,
                target,
                title=str(data.get("anchor_texts", "")),
                label="",
                color=edge_color,
            )

        network.write_html(str(output_path), notebook=False)

    def write_networkx_png(self, output_path: Path) -> None:
        if plt is None:
            return

        figure_width = max(12, min(24, self.graph.number_of_nodes() * 0.45))
        figure_height = max(8, min(18, self.graph.number_of_nodes() * 0.35))
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))

        layout_graph = self.graph.to_undirected()
        positions = nx.spring_layout(layout_graph, seed=42, k=None)

        dead_nodes = [node for node, data in self.graph.nodes(data=True) if data.get("dead")]
        start_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("start") and not data.get("dead")
        ]
        internal_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("internal") and not data.get("dead") and not data.get("start")
        ]
        external_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if not data.get("internal") and not data.get("dead") and not data.get("start")
        ]

        nx.draw_networkx_edges(
            self.graph,
            positions,
            ax=axis,
            edge_color="#9aa4ad",
            alpha=0.45,
            arrows=True,
            arrowsize=12,
            width=1.1,
        )
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=internal_nodes,
            node_color=INTERNAL_NODE_COLOR,
            node_size=650,
            ax=axis,
        )
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=external_nodes,
            node_color=EXTERNAL_NODE_COLOR,
            node_size=520,
            ax=axis,
        )
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=start_nodes,
            node_color=START_NODE_COLOR,
            node_size=980,
            ax=axis,
        )
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=dead_nodes,
            node_color=DEAD_NODE_COLOR,
            node_size=820,
            ax=axis,
        )

        labels = {
            node: str(data.get("label", guess_label(node)))[:40]
            for node, data in self.graph.nodes(data=True)
        }
        nx.draw_networkx_labels(
            self.graph,
            positions,
            labels=labels,
            font_size=8,
            font_color="#222222",
            ax=axis,
        )

        axis.set_title("Policy Link Graph (NetworkX)", fontsize=14)
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlsplit(url)
    cleaned = parsed._replace(fragment="")
    return urlunsplit(cleaned)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def slugify_url(url: str) -> str:
    parsed = urlsplit(url)
    candidate = f"{parsed.netloc}{parsed.path}".strip("/")
    candidate = re.sub(r"[^a-zA-Z0-9]+", "-", candidate).strip("-")
    return candidate or "policy-map"

def merge_pipe_separated(existing: str, new_value: str) -> str:
    values: list[str] = []
    for item in [existing, new_value]:
        for token in item.split(" | "):
            cleaned = clean_text(token)
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return " | ".join(values)

def looks_generic(label: str) -> bool:
    return is_generic_anchor_text(label)

def is_generic_anchor_text(text: str) -> bool:
    cleaned = clean_text(text).lower()
    if not cleaned:
        return True

    exact_matches = {
        "link",
        "here",
        "more",
        "read more",
        "learn more",
        "website",
        "site",
        "web site",
        "visit website",
        "visit site",
        "click here",
        "more info",
        "more information",
    }
    if cleaned in exact_matches:
        return True

    if "website" in cleaned or "web site" in cleaned:
        return True
    if cleaned.startswith("learn more"):
        return True
    if cleaned.startswith("read more"):
        return True
    return False

def site_label_from_url(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if not host:
        return url
    if host.startswith("www."):
        host = host[4:]
    return host

def extract_site_name(soup: BeautifulSoup, title: str = "") -> str:
    meta_candidates = [
        soup.find("meta", attrs={"property": "og:site_name"}),
        soup.find("meta", attrs={"name": "application-name"}),
        soup.find("meta", attrs={"name": "apple-mobile-web-app-title"}),
    ]
    for meta in meta_candidates:
        if meta and meta.get("content"):
            content = clean_text(meta.get("content", ""))
            if content:
                return content

    return extract_site_name_from_title(title)

def extract_site_name_from_title(title: str) -> str:
    cleaned = clean_text(title)
    if not cleaned:
        return ""

    parts = [part.strip() for part in re.split(r"\s+\|\s+|\s+[-—:]\s+", cleaned) if part.strip()]
    deduped_parts: list[str] = []
    for part in parts:
        if part not in deduped_parts:
            deduped_parts.append(part)

    if len(deduped_parts) >= 2:
        site_parts = deduped_parts[1:]
        meaningful = [part for part in site_parts if len(part) >= 4]
        if meaningful:
            return max(meaningful, key=lambda part: (len(part.split()), len(part)))

    return cleaned

def extract_page_label_from_title(title: str) -> str:
    cleaned = clean_text(title)
    if not cleaned:
        return ""

    parts = [part.strip() for part in re.split(r"\s+\|\s+|\s+[-—:]\s+", cleaned) if part.strip()]
    if not parts:
        return cleaned

    primary = parts[0]
    if len(primary) >= 3:
        return primary
    return cleaned

def guess_label(url: str, fallback: str = "") -> str:
    fallback_text = clean_text(fallback)
    if fallback_text and not is_generic_anchor_text(fallback_text):
        return fallback_text[:120]
    if fallback_text:
        return site_label_from_url(url)[:120]

    parsed = urlsplit(url)
    if parsed.path and parsed.path != "/":
        candidate = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        candidate = candidate.replace("-", " ").replace("_", " ").strip()
        if candidate:
            return candidate[:120]
    return parsed.netloc or url

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively crawl links in the main content area of a page, "
            "build a NetworkX graph, and flag dead links."
        )
    )
    parser.add_argument("url", help="Starting page to crawl.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for graph outputs. Default: %(default)s",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum recursive depth to follow internal links. Default: %(default)s",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum number of pages to fully crawl. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay between page fetches in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        default=[],
        help=(
            "Domain allowed for recursive crawling. Repeat this flag to add more. "
            "Defaults to the starting URL hostname."
        ),
    )
    parser.add_argument(
        "--content-selector",
        action="append",
        default=[],
        help=(
            "Preferred CSS selector for the main content area. "
            "Repeat this flag to try multiple selectors."
        ),
    )
    parser.add_argument(
        "--exclude-selector",
        action="append",
        default=[],
        help=(
            "CSS selector to strip from each page before link extraction. "
            "Repeat this flag to add more exclusions."
        ),
    )
    parser.add_argument(
        "--follow-external",
        action="store_true",
        help="Recursively follow external domains too. Default is off.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification.",
    )
    return parser

def make_config(args: argparse.Namespace) -> CrawlConfig:
    start_url = normalize_url(args.url)
    start_host = urlsplit(start_url).netloc.lower()
    allowed_domains = tuple(clean_text(domain).lower() for domain in args.allowed_domain if clean_text(domain))
    if not allowed_domains:
        allowed_domains = (start_host,)

    content_selectors = tuple(dict.fromkeys([*args.content_selector, *DEFAULT_CONTENT_SELECTORS]))
    exclude_selectors = tuple(dict.fromkeys([*DEFAULT_EXCLUDE_SELECTORS, *args.exclude_selector]))

    return CrawlConfig(
        start_url=start_url,
        output_dir=Path(args.output_dir),
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        timeout=args.timeout,
        delay=args.delay,
        allowed_domains=allowed_domains,
        content_selectors=content_selectors,
        exclude_selectors=exclude_selectors,
        follow_external=args.follow_external,
        verify_ssl=not args.insecure,
    )

def print_summary(summary: dict[str, object]) -> None:
    print(f"Pages crawled: {summary['pages_crawled']}")
    print(f"Nodes: {summary['nodes']}")
    print(f"Edges: {summary['edges']}")
    print(f"Dead nodes: {summary['dead_nodes']}")
    print(f"GraphML: {summary['graphml']}")
    print(f"NetworkX pickle: {summary['networkx_pickle']}")
    if summary.get("networkx_png"):
        print(f"NetworkX PNG: {summary['networkx_png']}")
    if summary.get("networkx_png_note"):
        print(summary["networkx_png_note"])
    print(f"Nodes CSV: {summary['nodes_csv']}")
    print(f"Edges CSV: {summary['edges_csv']}")
    if summary.get("html_graph"):
        print(f"HTML graph: {summary['html_graph']}")
    if summary.get("html_graph_note"):
        print(summary["html_graph_note"])
    print(f"Summary JSON: {summary['summary_json']}")

def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = make_config(args)
    crawler = PolicyCrawler(config)
    summary = crawler.crawl()
    print_summary(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
