PAPER    := paper
DIST     := $(PAPER)_arxiv
DIST_DIR := build/$(DIST)

.PHONY: all clean distclean view zip

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib arxiv.sty tables/embodied_carbon_table.tex figures/asymptotic_curves.png figures/asymptotic_ratio.png
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

view: $(PAPER).pdf
	open $(PAPER).pdf

# Build an arxiv-ready zip. arxiv compiles the source on its end, so the
# archive must include the .tex, the .bbl (so arxiv does not need a .bib),
# the style file, the tables/, and the figures/. The .bbl is generated
# from the local build.
zip: $(PAPER).pdf
	rm -rf $(DIST_DIR) $(DIST).zip
	mkdir -p $(DIST_DIR)/tables $(DIST_DIR)/figures
	cp $(PAPER).tex $(DIST_DIR)/
	cp arxiv.sty    $(DIST_DIR)/
	cp $(PAPER).bbl $(DIST_DIR)/
	cp tables/*.tex $(DIST_DIR)/tables/
	cp figures/*.png $(DIST_DIR)/figures/
	cd build && zip -r ../$(DIST).zip $(DIST)
	@echo "arxiv distribution written to $(DIST).zip"
	@unzip -l $(DIST).zip

clean:
	latexmk -c
	rm -f *.bbl *.blg *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -rf build $(DIST).zip
	rm -f $(PAPER).pdf
