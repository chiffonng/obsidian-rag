import re
from pathlib import Path
from typing import Dict, List, Any

from llama_index.schema import Document

DATA_DIR = Path("./data/raw")

kwargs = {"remove_heading_tags": True, "remove_math_code": False}


def clean_markdown_file(markdown_file: str | Path = None, **kwargs) -> str | None:
    """Turns a markdown file into plain text

    Args:
        markdown_file (str, optional): Defaults to None.
            Markdown file path
        **kwargs
            passed to strip_markdown_elements function

    Returns:
        str: Plain text
    """
    with open(markdown_file, "r") as file:
        markdown_content = file.read()
    return strip_markdown_elements(markdown_content, **kwargs)


def clean_markdown_dir(
    markdown_dir: str = None, return_documents: bool = False
) -> List[str] | Document:
    """Construct a database of clean text from
    a directory of markdown files

    Args:
        markdown_dir (str, optional):
            Markdown directory path, default: markdown_dir
        return_documents (bool, optional):
            Whether to return a list of llama_index.schema.Document; by default, return a list of clean text
    Returns:
        data (List[Document] | List[str]):
            List of Document objects from LlamaIndex;
            or list of clean text (depending on return_documents)
    """
    # get all markdown files in the directory and subdirectories
    files = list(Path(markdown_dir).rglob("*.md"))

    # load markdown files, and metadata if required
    data = []
    for file in files:
        text = clean_markdown_file(file)
        if text.strip():
            if not return_documents:
                data.append(text)
            else:
                # https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html
                data.append(
                    Document(
                        text=text,
                        metadata={
                            "file_name": file.stem,
                            "category": file.parent.name,
                        },
                    )
                )

    return data


def strip_markdown_elements(markdown_content: str, **kwargs) -> str | None:
    """Strips Markdown elements from a string of Markdown text.

    Args:
        markdown_content (str): String of Markdown text.
    Returns:
        clean_text (str / None): String of text with Markdown elements removed.
    """

    if not markdown_content.strip():
        return None

    # Extract keyword arguments
    remove_heading_tags = kwargs.get("remove_heading_tags", False)
    remove_math_code = kwargs.get("remove_math_code", False)

    # Define regular expressions for various Markdown elements to remove
    removed_patterns = [
        r"(-\s\[(.+?)\].*)",  # Checkboxes
        r"(-{3}(.*?)-{3}\s)",  # Front matter
        r"\|([^\n]*)\|\s*",  # Tables
        r"(\>\s+\[(.*?)[-+])|>\s",  # Callout flags
        r"((?<!\S)\#\w+\s)",  # Tags only
        r"\^.*?\s",  # Block identifiers e.g. ^eg006
        r"(\[\[)|(\]\])|(\!\[\[)\s",  # Backlink wrappers
        r"(\[(.*?)\]\((.*?)\))|(\(\[(.*?)\]\((.*?)\)\))\s",  # Links
        r"\!\[(.*?)\]\((.*?)\)\s",  # Images
        r"[\*\~\^\=\_\>\`]|(\s\-)",  # Markdown formatting
    ]

    if remove_heading_tags:
        removed_patterns.append(r"\#{1,6}\s")
    if remove_math_code:
        removed_patterns.insert(1, r"\$\$(.*?)\$\$\s")
        removed_patterns.insert(2, r"\$(.*?)\$\s")
        removed_patterns.insert(
            3, r"(`{3}[^`]+`{3})"
        )  # Code blocks, after front matter

    alias_pattern = r"\[\[(.*?)\|([^\]]*)\]\]"

    # Combine all patterns into a single pattern
    combined_pattern = "|".join(removed_patterns)

    # Capture backlinks with aliases and replace with purely aliases
    markdown_content = re.sub(alias_pattern, r"\2", markdown_content)

    # Remove Markdown elements using the combined pattern, delete the space left behind
    stripped_markdown = re.sub(combined_pattern, "", markdown_content, flags=re.DOTALL)

    # Remove empty lines
    clean_text = re.sub(r"\n\s*\n", "\n", stripped_markdown)

    return clean_text
