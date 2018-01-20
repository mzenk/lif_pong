#!/usr/bin/env bash
VIDNAME=$1

cd /home/hd/hd_hd/hd_kq433/git_repos/lif_pong/sampling/figures/general_animation
ls | grep "${VIDNAME}_[0-9]" > tmp.txt &&
sed -i -e 's/^/file /' tmp.txt &&
ffmpeg -f concat -i tmp.txt -c copy "${VIDNAME}_all.mp4" &&
rm tmp.txt