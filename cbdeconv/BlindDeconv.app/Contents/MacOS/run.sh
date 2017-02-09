#!/bin/sh
echo `pwd`> /Users/mjkoskin/cbd.txt
export DYLIB_LOAD_PATH=/Applications/BlindDeconv.app/Contents/Resources/libs
exec /Applications/BlindDeconv.app/Contents/Resources/cbd
exit 0
