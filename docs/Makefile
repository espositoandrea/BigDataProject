PDFS: data-exploration.pdf

all: $(PDFS)

%.pdf: %.ms
	groff -Tpdf -mspdf -kKutf8 $< >$@
