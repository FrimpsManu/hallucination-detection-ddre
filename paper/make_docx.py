#!/usr/bin/env python3
"""Build a Word version of the paper laid out like an ACL paper.

    python paper/make_docx.py OUT.docx

**Pipeline:**
1. Tectonic compiles ``main.tex`` (keeping ``main.aux``), so that equation,
   section, table and figure numbers come from LaTeX itself.
2. A copy of the source is simplified for pandoc: resolved ``\\ref``,
   numbered display equations, run-in paragraph heads and PNG figures.
3. pandoc converts it with the ACL citation style.
4. The .docx is rewritten to ACL geometry and typography.

**ACL geometry and typography (acl.sty):**
- A4 page, 2.5 cm margins.
- Title and authors across the page; the body in two columns, 7.7 cm wide
  with a 0.6 cm gap.
- Times New Roman: 11 pt body, 15 pt bold title, 12 pt author block.
- 10 pt abstract, indented 0.6 cm on both sides.
- Bold headings: 12 pt sections, 11 pt subsections.
- 10 pt captions and references; page numbers centred in the footer.

The LaTeX PDF remains the authoritative version; this file is for reading and
commenting. Requires tectonic, pandoc and rsvg-convert.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSL_URL = ("https://raw.githubusercontent.com/citation-style-language/styles/master/"
           "association-for-computational-linguistics.csl")

CM = 567  # twips per cm
PAGE_W, PAGE_H, MARGIN, GAP = 11906, 16838, round(2.5 * CM), round(0.6 * CM)
COLUMN_EMU = int(7.7 * 360000)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------- LaTeX side


def labels(aux_text):
    """{label: number} from ``main.aux`` (\\newlabel{key}{{num}{page}...})."""
    return {k: v for k, v in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux_text)}


def simplify(tex, numbers):
    def ref(m):
        key = m.group(1)
        if key not in numbers:
            raise SystemExit(f"unresolved \\ref{{{key}}}")
        return numbers[key]

    tex = re.sub(r"\\ref\{([^}]*)\}", ref, tex)

    def equation(m):
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", body)
        body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
        tag = f" \\qquad ({numbers[lab.group(1)]})" if lab else ""
        return f"$$ {body}{tag} $$"

    tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", equation, tex, flags=re.S)
    tex = re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1} ", tex)
    tex = re.sub(r"(figures/[A-Za-z0-9_]+)\.pdf", r"\1.png", tex)
    return tex


# ---------------------------------------------------------------- styles


def rpr(size_pt, bold=False, italic=False):
    return (f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
            f'w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
            f'{"<w:b/><w:bCs/>" if bold else ""}{"<w:i/>" if italic else ""}'
            f'<w:color w:val="000000"/><w:sz w:val="{int(size_pt * 2)}"/>'
            f'<w:szCs w:val="{int(size_pt * 2)}"/></w:rPr>')


def style(sid, name, ppr, run, kind="paragraph", based="Normal"):
    based_xml = f'<w:basedOn w:val="{based}"/>' if based and sid != "Normal" else ""
    return (f'<w:style w:type="{kind}" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'{based_xml}<w:qFormat/><w:pPr>{ppr}</w:pPr>{run}</w:style>')


def spacing(before=0, after=0, line=240):
    return f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'


STYLES = [
    style("Normal", "Normal", spacing(0, 0) + '<w:jc w:val="both"/>', rpr(11), based=None),
    style("BodyText", "Body Text", spacing(0, 0) + '<w:ind w:firstLine="227"/><w:jc w:val="both"/>',
          rpr(11)),
    style("FirstParagraph", "First Paragraph", spacing(0, 0) + '<w:jc w:val="both"/>', rpr(11)),
    style("Compact", "Compact", spacing(20, 20) + '<w:jc w:val="both"/>', rpr(11)),
    style("Title", "Title", spacing(0, 240) + '<w:jc w:val="center"/>', rpr(15, bold=True)),
    style("Author", "Author", spacing(0, 360) + '<w:jc w:val="center"/>', rpr(12)),
    style("AbstractTitle", "Abstract Title", spacing(0, 120) + '<w:jc w:val="center"/>',
          rpr(12, bold=True)),
    style("Abstract", "Abstract",
          spacing(0, 240) + f'<w:ind w:left="{GAP}" w:right="{GAP}"/><w:jc w:val="both"/>',
          rpr(10)),
    style("Heading1", "heading 1", spacing(240, 120) + '<w:keepNext/><w:jc w:val="left"/>'
          '<w:outlineLvl w:val="0"/>', rpr(12, bold=True)),
    style("Heading2", "heading 2", spacing(200, 100) + '<w:keepNext/><w:jc w:val="left"/>'
          '<w:outlineLvl w:val="1"/>', rpr(11, bold=True)),
    style("Heading3", "heading 3", spacing(160, 80) + '<w:keepNext/><w:jc w:val="left"/>'
          '<w:outlineLvl w:val="2"/>', rpr(11, bold=True)),
    style("Caption", "caption", spacing(120, 200) + '<w:jc w:val="both"/>', rpr(10)),
    style("ImageCaption", "Image Caption", spacing(80, 200) + '<w:jc w:val="both"/>', rpr(10),
          based="Caption"),
    style("TableCaption", "Table Caption", spacing(120, 80) + '<w:jc w:val="both"/>', rpr(10),
          based="Caption"),
    style("CaptionedFigure", "Captioned Figure", spacing(120, 0) + '<w:jc w:val="center"/>',
          rpr(11)),
    style("Figure", "Figure", spacing(120, 0) + '<w:jc w:val="center"/>', rpr(11)),
    style("Bibliography", "Bibliography",
          spacing(0, 100) + '<w:ind w:left="227" w:hanging="227"/><w:jc w:val="both"/>', rpr(10)),
]

TABLE_STYLE = (
    '<w:style w:type="table" w:styleId="Table"><w:name w:val="Table"/>'
    '<w:pPr><w:spacing w:before="20" w:after="20"/><w:jc w:val="center"/></w:pPr>'
    + rpr(9) +
    '<w:tblPr><w:jc w:val="center"/><w:tblBorders>'
    '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
    '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
    '</w:tblBorders><w:tblCellMar><w:left w:w="28" w:type="dxa"/>'
    '<w:right w:w="28" w:type="dxa"/></w:tblCellMar></w:tblPr>'
    '<w:tblStylePr w:type="firstRow"><w:rPr><w:b w:val="0"/></w:rPr><w:tcPr><w:tcBorders>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders></w:tcPr>'
    '</w:tblStylePr></w:style>')


def restyle(styles_xml):
    ids = re.compile(r'<w:style\b[^>]*w:styleId="([^"]+)".*?</w:style>', re.S)
    replaced = {re.search(r'w:styleId="([^"]+)"', s).group(1) for s in STYLES} | {"Table"}
    styles_xml = ids.sub(lambda m: "" if m.group(1) in replaced else m.group(0), styles_xml)
    defaults = ('<w:docDefaults><w:rPrDefault>' + rpr(11) + '</w:rPrDefault>'
                '<w:pPrDefault><w:pPr>' + spacing(0, 0) + '</w:pPr></w:pPrDefault>'
                '</w:docDefaults>')
    styles_xml = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", defaults, styles_xml, flags=re.S)
    return styles_xml.replace("</w:styles>", "".join(STYLES) + TABLE_STYLE + "</w:styles>")


# ---------------------------------------------------------------- document


def section(cols, footer_rid, continuous):
    kind = '<w:type w:val="continuous"/>' if continuous else ""
    footer = (f'<w:footerReference w:type="default" r:id="{footer_rid}"/>'
              if footer_rid else "")
    return (f'<w:sectPr>{footer}{kind}'
            f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
            f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" '
            f'w:left="{MARGIN}" w:header="708" w:footer="567" w:gutter="0"/>'
            f'<w:cols w:num="{cols}" w:space="{GAP}"/></w:sectPr>')


COLUMN_TWIPS = round(7.7 * CM)


def fit_table(m):
    """Fixed column widths that fill the 7.7 cm column; the label column wider.
    Cell widths honour gridSpan; tcW is written first in tcPr (schema order)."""
    tbl = m.group(0)
    n = len(re.findall(r"<w:gridCol\b", tbl))
    if n == 0:
        return tbl
    first = round(COLUMN_TWIPS * (0.28 if n > 2 else 0.5))
    rest = (COLUMN_TWIPS - first) // max(n - 1, 1)
    widths = [first] + [rest] * (n - 1)
    tbl = re.sub(r"<w:tblW [^/]*/>", f'<w:tblW w:w="{COLUMN_TWIPS}" w:type="dxa"/>'
                 '<w:tblLayout w:type="fixed"/>', tbl, count=1)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", f"<w:tblGrid>{grid}</w:tblGrid>", tbl, flags=re.S)

    def row(rm):
        pos = [0]

        def cell(cm):
            tc = cm.group(0)
            span = re.search(r'<w:gridSpan w:val="(\d+)"', tc)
            k = int(span.group(1)) if span else 1
            w = sum(widths[pos[0]:pos[0] + k])
            pos[0] += k
            extra = f'<w:gridSpan w:val="{k}"/>' if span else ""
            return re.sub(r"<w:tcPr\s*/>|<w:tcPr>.*?</w:tcPr>",
                          f'<w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{extra}</w:tcPr>',
                          tc, count=1, flags=re.S)

        return re.sub(r"<w:tc>.*?</w:tc>", cell, rm.group(0), flags=re.S)

    return re.sub(r"<w:tr\b.*?</w:tr>", row, tbl, flags=re.S)


def relayout(doc, footer_rid):
    # Title and author block in one column, then a continuous section break
    # into two columns: the layout of ACL's own Word template. (Microsoft Word
    # and Google Docs keep both on page 1; Apple Pages starts a new page at
    # any section break.)
    authors = list(re.finditer(r'<w:p>(?:(?!</w:p>).)*?w:val="Author".*?</w:p>', doc, re.S))
    if not authors:
        raise SystemExit("no Author paragraph found")
    last = authors[-1]
    brk = ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
           + section(1, footer_rid, True) + '</w:pPr></w:p>')
    doc = doc[:last.end()] + brk + doc[last.end():]
    doc = re.sub(r"<w:sectPr\b(?:(?!<w:sectPr\b).)*?</w:sectPr>\s*</w:body>",
                 section(2, footer_rid, True) + "</w:body>", doc, flags=re.S)
    # Figures: scale to the column width, keeping the aspect ratio.

    def fit(m):
        cx, cy = int(m.group(1)), int(m.group(2))
        return f'cx="{COLUMN_EMU}" cy="{int(cy * COLUMN_EMU / cx)}"'

    doc = re.sub(r'cx="(\d+)" cy="(\d+)"', fit, doc)
    # Tables: explicit widths filling the column, first (label) column wider.
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", fit_table, doc, flags=re.S)
    # Paragraphs with inline math: left-aligned, since some renderers stretch
    # the letters of justified lines that contain equations.

    def left(m):
        p = m.group(0)
        if "<m:oMath>" not in p or "<m:oMathPara>" in p or "<w:jc " in p:
            return p
        if re.search(r"<w:pStyle [^>]*/>", p):
            return re.sub(r"(<w:pStyle [^>]*/>)", r'\1<w:jc w:val="left"/>', p, count=1)
        if "<w:pPr>" in p:
            return p.replace("<w:pPr>", '<w:pPr><w:jc w:val="left"/>', 1)
        return p.replace("<w:p>", '<w:p><w:pPr><w:jc w:val="left"/></w:pPr>', 1)

    doc = re.sub(r"<w:p>(?:(?!</w:p>).)*?</w:p>", left, doc, flags=re.S)
    return doc


FOOTER = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          f'<w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          f'<w:r>{rpr(10)}<w:fldChar w:fldCharType="begin"/></w:r>'
          f'<w:r>{rpr(10)}<w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          f'<w:r>{rpr(10)}<w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def rewrite(src, dst):
    with zipfile.ZipFile(src) as zin:
        files = {n: zin.read(n) for n in zin.namelist()}
    rid = "rIdAclFooter"
    rels = files["word/_rels/document.xml.rels"].decode()
    rels = rels.replace("</Relationships>",
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                        'officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
                        "</Relationships>")
    ctypes = files["[Content_Types].xml"].decode()
    if "footer1.xml" not in ctypes:
        ctypes = ctypes.replace("</Types>", '<Override PartName="/word/footer1.xml" '
                                'ContentType="application/vnd.openxmlformats-officedocument.'
                                'wordprocessingml.footer+xml"/></Types>')
    files["word/_rels/document.xml.rels"] = rels.encode()
    files["[Content_Types].xml"] = ctypes.encode()
    files["word/footer1.xml"] = FOOTER.encode()
    files["word/styles.xml"] = restyle(files["word/styles.xml"].decode()).encode()
    files["word/document.xml"] = relayout(files["word/document.xml"].decode(), rid).encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in ["[Content_Types].xml"] + [n for n in files if n != "[Content_Types].xml"]:
            zout.writestr(name, files[name])


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "paper.docx").expanduser().resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("main.tex", "refs.bib", "acl.sty", "acl_natbib.bst"):
            shutil.copy(HERE / name, tmp / name)
        shutil.copytree(HERE / "figures", tmp / "figures")
        subprocess.run(["tectonic", "--keep-intermediates", "main.tex"], cwd=tmp, check=True,
                       capture_output=True)
        numbers = labels((tmp / "main.aux").read_text())
        for svg in (ROOT / "docs" / "figures").glob("*.svg"):
            subprocess.run(["rsvg-convert", "-f", "png", "-z", "2", "-o",
                            str(tmp / "figures" / f"{svg.stem}.png"), str(svg)], check=True)
        (tmp / "pandoc.tex").write_text(simplify((tmp / "main.tex").read_text(), numbers))
        csl = tmp / "acl.csl"
        subprocess.run(["curl", "-sSLf", "-o", str(csl), CSL_URL], check=True)
        subprocess.run(["pandoc", "pandoc.tex", "-o", "raw.docx", "--citeproc",
                        "--bibliography", "refs.bib", "--csl", str(csl),
                        "-M", "reference-section-title=References", "--number-sections",
                        "--resource-path", "."], cwd=tmp, check=True)
        rewrite(tmp / "raw.docx", out)
    print(out)


if __name__ == "__main__":
    main()
