#!/bin/bash

youtube-dl $1 -o - | ffmpeg -i - -f wav - | pv | python sentiment-vsm-ensemble.py

