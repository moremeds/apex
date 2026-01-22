#!/usr/bin/env python3
"""Markdown to PDF converter with compact styling and Chinese font support."""

import argparse
import markdown
from pathlib import Path
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

SCRIPT_DIR = Path(__file__).parent


def get_css_style() -> str:
    """Generate CSS - uses system fonts via fontconfig."""
    return """
@page {
    size: A4;
    margin: 1.5cm 1.5cm 1.5cm 1.5cm;
}

body {
    font-family: "Heiti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #333;
}

h1 {
    font-size: 18pt;
    color: #1a1a2e;
    border-bottom: 2px solid #4a4e69;
    padding-bottom: 4px;
    margin: 16px 0 10px 0;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    color: #22223b;
    border-bottom: 1px solid #9a8c98;
    padding-bottom: 3px;
    margin: 12px 0 8px 0;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    color: #4a4e69;
    margin: 10px 0 6px 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    color: #666;
    margin: 8px 0 4px 0;
}

p {
    margin: 4px 0;
    text-align: justify;
}

pre {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 8px;
    font-family: "SF Mono", "Menlo", "Monaco", monospace;
    font-size: 8.5pt;
    line-height: 1.3;
    overflow-x: auto;
    margin: 6px 0;
    page-break-inside: avoid;
}

code {
    background-color: #f0f0f0;
    padding: 1px 4px;
    border-radius: 2px;
    font-family: "SF Mono", "Menlo", "Monaco", monospace;
    font-size: 9pt;
}

pre code {
    background-color: transparent;
    padding: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 9pt;
}

th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

ul, ol {
    margin: 4px 0;
    padding-left: 20px;
}

li {
    margin: 2px 0;
}

blockquote {
    border-left: 3px solid #4a4e69;
    margin: 6px 0;
    padding: 4px 12px;
    background-color: #f9f9f9;
    color: #555;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 10px 0;
}

strong {
    color: #1a1a2e;
}

/* Section styling */
.question {
    background-color: #f8f9fa;
    border-left: 3px solid #4a4e69;
    padding: 6px 10px;
    margin: 8px 0 4px 0;
}

.answer {
    margin-left: 10px;
}

/* Syntax highlighting */
.highlight .k { color: #0000ff; }  /* Keyword */
.highlight .s { color: #a31515; }  /* String */
.highlight .c { color: #008000; }  /* Comment */
.highlight .n { color: #000000; }  /* Name */
.highlight .o { color: #666666; }  /* Operator */
.highlight .p { color: #000000; }  /* Punctuation */
"""


def convert_md_to_pdf(input_path: str, output_path: str | None = None) -> Path:
    """Convert a Markdown file to PDF."""
    input_file = Path(input_path)

    if output_path is None:
        output_path = input_file.with_suffix('.pdf')
    else:
        output_path = Path(output_path)

    # Read markdown content
    md_content = input_file.read_text(encoding='utf-8')

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'fenced_code',
            'tables',
            'codehilite',
            'toc',
            'nl2br',
        ],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': True,
            }
        }
    )

    # Wrap in HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{input_file.stem}</title>
</head>
<body>
{html_content}
</body>
</html>
"""

    # Configure fonts
    font_config = FontConfiguration()

    # Convert to PDF
    html = HTML(string=full_html, base_url=str(input_file.parent))
    css = CSS(string=get_css_style(), font_config=font_config)

    html.write_pdf(str(output_path), stylesheets=[css], font_config=font_config)
    print(f"Generated: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF')
    parser.add_argument('input', help='Input Markdown file or directory')
    parser.add_argument('-o', '--output', help='Output PDF file (optional)')
    parser.add_argument('--all', action='store_true', help='Convert all .md files in directory')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.all and input_path.is_dir():
        # Convert all markdown files in directory
        for md_file in sorted(input_path.glob('*.md')):
            convert_md_to_pdf(str(md_file))
    elif input_path.is_file():
        convert_md_to_pdf(str(input_path), args.output)
    else:
        print(f"Error: {input_path} not found")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
