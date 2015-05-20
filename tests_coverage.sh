#!/bin/bash
echo 'Determine coverage of all unittests...'
nosetests-2.7 --with-coverage \
    --cover-erase --cover-inclusive \
    --cover-package=binary_response \
    --cover-html --cover-html-dir="docs/coverage"
    