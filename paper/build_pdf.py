import base64
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "PAPER.md")
HTML = os.path.join(HERE, "PAPER.html")
PDF = os.path.join(HERE, "PAPER.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.42; color: #0b0b0b; max-width: 176mm; margin: 0 auto; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #e4e3df; padding-bottom: 2pt; }
p { margin: 5pt 0; text-align: justify; }
p.subtitle { font-style: italic; color: #52514e; margin-bottom: 10pt; }
p.byline { color: #52514e; margin-bottom: 6pt; }
ul { margin: 3pt 0 5pt 18pt; padding: 0; }
li { margin: 2pt 0; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.2pt; background: #f2f1ee; padding: 0 2px; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 9.6pt; }
th, td { border-bottom: 1px solid #e4e3df; padding: 2.5pt 7pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: #52514e; font-weight: 600; }
figure { margin: 8pt auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 95mm; }
figcaption { font-size: 9pt; color: #52514e; margin-top: 2pt; }
"""


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<em>\1</em>", text)
    return text


def image_tag(alt, src):
    path = os.path.join(HERE, src)
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    out = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def convert(md):
    lines = md.splitlines()
    out, i, first_h1 = [], 0, True
    while i < len(lines):
        l = lines[i]
        if l.startswith("# "):
            out.append(f"<h1>{inline(l[2:])}</h1>")
            i += 1
            if first_h1 and i < len(lines) and lines[i].strip() == "" and i + 1 < len(lines) and lines[i + 1].startswith("*"):
                out.append(f'<p class="subtitle">{inline(lines[i + 1].strip("*"))}</p>')
                i += 2
            first_h1 = False
        elif l.startswith("## "):
            out.append(f"<h2>{inline(l[3:])}</h2>")
            i += 1
        elif l.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", l)
            out.append(image_tag(m.group(1), m.group(2)))
            i += 1
        elif l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
        elif l.startswith("- "):
            j = i
            items = []
            while j < len(lines) and lines[j].startswith("- "):
                items.append(f"<li>{inline(lines[j][2:])}</li>")
                j += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            i = j
        elif l.strip() == "":
            i += 1
        else:
            j = i
            para = []
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "!", "|", "- ")):
                para.append(lines[j])
                j += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
            i = j
    return "\n".join(out)


def main():
    with open(SRC) as fh:
        body = convert(fh.read())
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>PAPER</title><style>{CSS}</style></head><body>{body}</body></html>"
    with open(HTML, "w") as fh:
        fh.write(doc)
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", f"file://{HTML}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    os.remove(HTML)
    print(PDF, os.path.getsize(PDF), "bytes")


if __name__ == "__main__":
    sys.exit(main())
