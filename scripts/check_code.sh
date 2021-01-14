#!/bin/bash

set -e

err_report() {
    echo
    echo "/!\\ Code is not ready to be pushed /!\\"
    echo
}

trap 'err_report' ERR

echo "> Running isort"
isort . 
echo "> Running black"
black . 
echo "> Running flake8"
flake8

echo
echo "Code is ready to be pushed"
echo
