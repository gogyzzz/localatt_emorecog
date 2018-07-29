#!/bin/bash

osmdir=$1
myconfdir="opensmile_configs"
cp $myconfdir/*.conf.inc $osmdir/config/shared/
cp $myconfdir/*.conf $osmdir/config/
