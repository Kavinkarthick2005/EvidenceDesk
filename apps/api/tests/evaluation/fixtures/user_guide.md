# EvidenceDesk User Onboarding and Workflow Guide

## 1. Organizing Documents into Workspaces
Users can organize reference libraries into separate named workspace collections to isolate project files. Documents uploaded within a workspace remain segregated from other projects.

## 2. Inspecting Cited Evidence Passages
When an answer references a supporting claim, users can click on the citation tag to inspect the underlying passage text, extracted PDF page numbers, and enclosing Markdown headings.

## 3. Supported Text Encodings
The text extraction service automatically detects character encodings including UTF-8, UTF-16 with byte-order marks, and Windows-1252 character sets. Files with invalid binary null bytes are rejected.

## 4. Empty File Upload Handling
Uploading an empty 0-byte document triggers an immediate validation error before any storage write or embedding generation occurs. The client receives an informative error message.

## 5. PDF Page Number Extraction
PDF citations report 1-based page numbers extracted from the PDF document catalog. If a passage spans across a page boundary, all covered page numbers are recorded in the chunk metadata.
