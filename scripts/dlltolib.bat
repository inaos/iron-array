@echo off
for %%f in (%1) do set urlpath=%%~nf
dumpbin /EXPORTS %1 > %urlpath%.exports
python %2 %urlpath%.exports
lib /def:%urlpath%.def /machine:x64 /out:%3\%urlpath%.lib
