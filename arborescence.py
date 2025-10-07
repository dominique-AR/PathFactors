# visualize_tree_nx.py
from pathlib import Path
import re
import networkx as nx
import matplotlib.pyplot as plt

# === Réglages ===
ROOT = Path("D:/2025 Dominique/DOSSIER MEMOIRE/Ressources/Demo/traffic_dashboard").resolve()  # Explicit root path
IGNORE_PARTS = {".streamlit", ".git", ".venv", "__pycache__", "node_modules"}
MAX_DEPTH = 10

# Ordre forcé au niveau racine (0 = priorité)
PIN_ORDER = {
    ROOT: {"app.py": 0, "pages": 1, "utils": 2}
}

# === Tri naturel et spécial "pages" ===
_digit_re = re.compile(r"(\d+)")

def natural_key(s: str):
    return tuple(int(x) if x.isdigit() else x.lower() for x in _digit_re.split(s))

def extract_prefix_num(name: str):
    m = re.match(r"^(\d+)[^0-9]", name)
    return int(m.group(1)) if m else None

def is_ignored(p: Path) -> bool:
    return any(part in IGNORE_PARTS for part in p.parts)

def children_sorted(parent: Path):
    try:
        items = [c for c in parent.iterdir() if not is_ignored(c)]
    except (PermissionError, OSError):
        return []

    dirs = [c for c in items if c.is_dir()]
    files = [c for c in items if c.is_file()]

    if parent.name.lower() == "pages":
        def pages_key(p: Path):
            num = extract_prefix_num(p.name)
            return (0 if num is not None else 1,
                    num if num is not None else float("inf"),
                    natural_key(p.name))
        dirs.sort(key=pages_key)
        files.sort(key=pages_key)
    else:
        dirs.sort(key=lambda p: natural_key(p.name))
        files.sort(key=lambda p: natural_key(p.name))

    ordered = dirs + files

    pinmap = PIN_ORDER.get(parent, {})
    if pinmap:
        ordered.sort(key=lambda p: (pinmap.get(p.name, float('inf')), natural_key(p.name)))

    return ordered

# === Construction du graphe ===
def scan_tree(root: Path, max_depth=MAX_DEPTH):
    G = nx.DiGraph()
    node_levels = {}
    node_is_dir = {}

    def walk(p: Path, level: int):
        if max_depth is not None and level > max_depth:
            return
        try:
            G.add_node(str(p))
            node_levels[p] = level
            node_is_dir[p] = p.is_dir()
            if p.is_dir():
                for c in children_sorted(p):
                    G.add_node(str(c))
                    node_levels[c] = level + 1
                    node_is_dir[c] = c.is_dir()
                    G.add_edge(str(p), str(c))
                    walk(c, level + 1)
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot access {p} due to {e}")

    walk(root, 0)
    return G, node_levels, node_is_dir

# === Layout en arbre ===
def tree_layout(G: nx.DiGraph, root: Path, node_levels: dict, x_gap=2.0, y_gap=1.7):
    children = {n: [] for n in G.nodes}
    for u, v in G.edges:
        children[u].append(v)

    pos = {}
    next_x = [0]
    visited = set()

    def assign_x(n: str):
        if n in visited:
            return pos[n][0]
        visited.add(n)
        p = Path(n)
        if n == str(root):  # Ensure root is positioned first
            pos[n] = (next_x[0] * x_gap, -node_levels[p] * y_gap)
            next_x[0] += 1
            return pos[n][0]
        if len(children[n]) == 0:
            x = next_x[0]
            next_x[0] += 1
            pos[n] = (x * x_gap, -node_levels[p] * y_gap)
            return x
        xs = [assign_x(c) for c in children[n] if c in G.nodes]
        x = sum(xs) / len(xs) if xs else next_x[0]
        next_x[0] += 1  # Increment for parent node
        pos[n] = (x * x_gap, -node_levels[p] * y_gap)
        return x

    # Start with root to ensure it has a position
    assign_x(str(root))
    # Assign positions to any remaining unpositioned nodes
    for n in G.nodes:
        if n not in pos:
            print(f"Assigning default position to unpositioned node: {n}")
            p = Path(n)
            pos[n] = (next_x[0] * x_gap, -node_levels[p] * y_gap)
            next_x[0] += 1

    return pos

# === Exécution ===
if __name__ == "__main__":
    G, levels, is_dir = scan_tree(ROOT)
    pos = tree_layout(G, ROOT, levels)

    dir_nodes = [n for n, d in is_dir.items() if d]
    file_nodes = [n for n, d in is_dir.items() if not d]

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_edges(G, pos, arrows=False, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=dir_nodes, node_shape="s", node_size=1200,
                           node_color="lightblue", edgecolors="black")
    nx.draw_networkx_nodes(G, pos, nodelist=file_nodes, node_shape="o", node_size=900,
                           node_color="lightgreen", edgecolors="black")
    labels = {n: Path(n).name for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='normal')

    plt.title("Structure du projet (arborescence ordonnée)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("project_structure.png", dpi=200, bbox_inches='tight')
    plt.show()