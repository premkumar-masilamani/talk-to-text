OS := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH := $(shell uname -m | sed 's/x86_64/x64/')

# Default values
OS_NAME := $(OS)
SEP := :
EXE_EXT :=

ifeq ($(OS),darwin)
	OS_NAME := macos
else
	ifneq (,$(findstring mingw,$(OS)))
		OS_NAME := windows
		SEP := ;
		EXE_EXT := .exe
	endif
endif

.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	pipenv install

.PHONY: show
show:
	@echo "Listing dependencies..."
	pipenv graph

.PHONY: run
run:
	pipenv run python -m transcriber.main -v

.PHONY: ui
ui:
	pipenv run python -m transcriber.ui_app

.PHONY: build
build:
ifeq ($(OS),darwin)
	$(MAKE) build-mac
else ifneq (,$(findstring mingw,$(OS)))
	$(MAKE) build-windows
else
	$(MAKE) build-linux
endif

.PHONY: build-linux
build-linux:
	@echo "Building for linux-$(ARCH)..."
	rm -rf dist build
	pipenv run pyinstaller --noconfirm --clean --onefile --windowed --name talk-to-text \
		--add-data "files/talk-to-text-icon.png:files" \
		--icon "files/talk-to-text-icon.png" \
		transcriber/ui_app.py
	@echo "Packaging..."
	cd dist && zip -9 "talk-to-text-linux-$(ARCH).zip" talk-to-text

.PHONY: build-mac
build-mac:
	@echo "Building for macos-$(ARCH)..."
	rm -rf dist build
	pipenv run pyinstaller --noconfirm --clean --onefile --windowed --name talk-to-text \
		--add-data "files/talk-to-text-icon.png:files" \
		--icon "files/talk-to-text-icon.png" \
		transcriber/ui_app.py
	@echo "Packaging..."
	cd dist && zip -9 "talk-to-text-macos-$(ARCH).zip" talk-to-text

.PHONY: build-windows
build-windows:
	@echo "Building for windows-$(ARCH)..."
	rm -rf dist build
	pipenv run pyinstaller --noconfirm --clean --onefile --windowed --name talk-to-text \
		--add-data "files/talk-to-text-icon.png;files" \
		--icon "files/talk-to-text-icon.png" \
		transcriber/ui_app.py
	@echo "Packaging..."
	cd dist && powershell.exe -Command "Compress-Archive -Path talk-to-text.exe -DestinationPath talk-to-text-windows-$(ARCH).zip"
