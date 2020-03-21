@echo off
for %%f in (%1) do set filename=%%~nf
dumpbin /EXPORTS %1 > %filename%.exports
python %2 %filename%.exports
lib /def:%filename%.def /machine:x64 /out:%3\%filename%.lib
