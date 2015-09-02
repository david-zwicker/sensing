#!/bin/bash
echo 'Run unittests in parallel...'
nosetests-2.7 --processes=30 --process-timeout=20 --stop
