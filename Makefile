# Generic LaTeX Makefile (latexmk + pdflatex, out-of-tree build)
#
# Usage:
#   make               # build $(DOC).tex
#   make DOC=paper     # build paper.tex
#   make watch         # continuous rebuild on save
#   make clean         # remove aux files
#   make distclean     # remove the whole build dir
#
# VS Code's latex-workshop passes DOC=%DOCFILE% (filename without extension).

DOC       ?= main
BUILD_DIR ?= build
ENGINE    ?= -pdf          # use -lualatex or -xelatex if you need unicode/fontspec

LATEXMK      ?= latexmk
LATEXMKFLAGS ?= $(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error \
                -synctex=1 -outdir=$(BUILD_DIR)

DOCBASE := $(basename $(DOC))
PDF     := $(BUILD_DIR)/$(DOCBASE).pdf

.PHONY: all watch clean distclean
.DEFAULT_GOAL := all

all: $(PDF)

$(PDF): $(DOCBASE).tex $(wildcard *.tex) $(wildcard *.bib) $(wildcard *.sty) $(wildcard *.cls)
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMKFLAGS) $(DOCBASE).tex

watch:
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMKFLAGS) -pvc $(DOCBASE).tex

clean:
	$(LATEXMK) -c -outdir=$(BUILD_DIR) $(DOCBASE).tex

distclean:
	rm -rf $(BUILD_DIR)