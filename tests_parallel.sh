#!/bin/bash
echo 'Run unittests in parallel...'
nosetests-2.7 --processes=4 --process-timeout=60
