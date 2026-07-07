from __future__ import annotations

import csv
import re
import zipfile
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Iterable, Sequence


INVALID_XML_CHARS = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
)


def write_excel_csv(
    path: str | Path,
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
) -> None:
    """Write a standards-friendly CSV with a UTF-8 BOM and Excel separator hint."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_file.write("sep=;\r\n")
        writer = csv.writer(csv_file, delimiter=";", lineterminator="\r\n")
        writer.writerow(headers)
        writer.writerows(rows)


def write_xlsx(
    path: str | Path,
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
    *,
    sheet_name: str = "BPM Results",
) -> None:
    row_list = [list(headers), *[list(row) for row in rows]]
    row_count = max(1, len(row_list))
    col_count = max(1, max(len(row) for row in row_list))
    dimension = f"A1:{_cell_ref(row_count, col_count)}"
    safe_sheet_name = _clean_sheet_name(sheet_name)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", _content_types_xml())
        workbook.writestr("_rels/.rels", _root_rels_xml())
        workbook.writestr("docProps/app.xml", _app_xml())
        workbook.writestr("docProps/core.xml", _core_xml())
        workbook.writestr("xl/workbook.xml", _workbook_xml(safe_sheet_name))
        workbook.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        workbook.writestr("xl/styles.xml", _styles_xml())
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            _worksheet_xml(row_list, col_count, dimension),
        )


def _worksheet_xml(
    rows: Sequence[Sequence[object]],
    col_count: int,
    dimension: str,
) -> str:
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index in range(1, col_count + 1):
            value = row[col_index - 1] if col_index <= len(row) else ""
            cells.append(_cell_xml(row_index, col_index, value, is_header=row_index == 1))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    column_xml = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(_column_widths(col_count), start=1)
    )

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <cols>{column_xml}</cols>
  <sheetData>{"".join(xml_rows)}</sheetData>
  <autoFilter ref="{dimension}"/>
  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>"""


def _cell_xml(row_index: int, col_index: int, value: object, *, is_header: bool) -> str:
    ref = _cell_ref(row_index, col_index)
    style = ' s="1"' if is_header else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f'<c r="{ref}"{style}><v>{value}</v></c>'

    text = _clean_xml_text(str(value))
    space = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f'<c r="{ref}" t="inlineStr"{style}><is><t{space}>{escape(text)}</t></is></c>'


def _cell_ref(row_index: int, col_index: int) -> str:
    letters = ""
    index = col_index
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{row_index}"


def _column_widths(col_count: int) -> list[int]:
    defaults = [52, 12, 14, 18, 16, 42, 16, 16, 90]
    if col_count <= len(defaults):
        return defaults[:col_count]
    return defaults + [18] * (col_count - len(defaults))


def _clean_xml_text(value: str) -> str:
    return INVALID_XML_CHARS.sub("", value)


def _clean_sheet_name(value: str) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", value).strip()
    return cleaned[:31] or "Sheet1"


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""


def _root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _workbook_xml(sheet_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def _workbook_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>BPM Finder</Application>
</Properties>"""


def _core_xml() -> str:
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>BPM Finder</dc:creator>
  <cp:lastModifiedBy>BPM Finder</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>"""
