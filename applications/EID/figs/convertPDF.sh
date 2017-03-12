#!/bin/bash
for f in ./*.eps; do
    epspdf $f $(basename $f .eps).pdf
done
