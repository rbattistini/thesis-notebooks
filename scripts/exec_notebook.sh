#!/bin/sh
jupyter nbconvert --inplace --to notebook --execute "$@"
