#!/bin/bash
# Quantize Blackfrost DERISKED-FP8 to EXL3 ~2.9bpw (reference recipe class)
LOG=/home/raulwesche/blackfrost-quant.log
SRC=/home/raulwesche/models/blackfrost-fp8-src
OUT=/home/raulwesche/models/blackfrost-exl3-2.9bpw
WORK=/home/raulwesche/blackfrost-quant-work
PY=/home/raulwesche/venvs/exl3_v41/bin/python
export PYTHONPATH=/home/raulwesche/exllamav3

echo "[$(date)] waiting for download to finish..." > $LOG
while ! grep -q 'DOWNLOAD COMPLETE' /home/raulwesche/blackfrost-download.log 2>/dev/null; do
  # also bail if downloader died
  if ! pgrep -f 'snapshot_download' > /dev/null; then
    echo "[$(date)] downloader process gone, checking..." >> $LOG
    if grep -q 'DOWNLOAD COMPLETE' /home/raulwesche/blackfrost-download.log 2>/dev/null; then break; fi
    echo "[$(date)] DOWNLOADER DIED WITHOUT COMPLETING - ABORT" >> $LOG
    exit 1
  fi
  sleep 60
done
echo "[$(date)] download complete, starting quant" >> $LOG

mkdir -p $OUT $WORK
cd /home/raulwesche/exllamav3
$PY convert.py \
  -i $SRC \
  -w $WORK \
  -o $OUT \
  -b 3 -hq \
  -hb 6 -mb 4 -vb 16 -eb 16 -ngb 16 \
  -ss 8192 \
  -cd conversion/standard_cal_data \
  -v \
  >> $LOG 2>&1
RC=$?
echo "[$(date)] convert.py exit code: $RC" >> $LOG
if [ $RC -eq 0 ]; then echo 'QUANT COMPLETE' >> $LOG; else echo 'QUANT FAILED' >> $LOG; fi
