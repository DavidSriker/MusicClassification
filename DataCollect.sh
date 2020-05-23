#!/usr/bin/env bash

# script to download the data
unzip_from_link() {
  local link=$1
  local dir=$2

  curl -L -o "${dir}/tmp.zip" ${link}
  echo "unziping..."
  7z x "${dir}/tmp.zip" -o${dir}
  $(rm "${dir}/tmp.zip")
}

$(mkdir -p "Data/")
if ! [ -x "$(command -v 7z)" ]; then
  echo "Error: 7z {p7zip} is not installed"
  exit 1
fi
echo "Downloading Data"
unzip_from_link "https://os.unil.cloud.switch.ch/fma/fma_small.zip" "Data"
unzip_from_link "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" "Data"
