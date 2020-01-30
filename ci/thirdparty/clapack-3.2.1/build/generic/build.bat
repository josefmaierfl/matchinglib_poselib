@pushd \
@%~d0
@cd "%~dp0"

@if NOT "%1"=="" (@set THIRDPARTYPREFIX=%1) else (@call "%THIRDPARTYROOT%\_buildtools\genericSettings.bat")
@call "%THIRDPARTYROOT%\_buildtools\%THIRDPARTYPREFIX%BuildHelper.bat"

@mkdir ..\%THIRDPARTYPREFIX%\
::@mkdir ..\..\bin\%THIRDPARTYPREFIX%\
@mkdir ..\..\lib\%THIRDPARTYPREFIX%\
@cd ..\%THIRDPARTYPREFIX%\


@echo :::::: DEBUG BUILD ::::::
@mkdir debug
@cd debug

@cmake -G "%THIRDPARTYCMAKESTR%" ../../../

@if not "%THIRDPARTYCOMPILER%"=="vs10" (@"%DEVENV%" CLAPACK.sln /build Debug /project ALL_BUILD /projectconfig Debug)
@if "%THIRDPARTYCOMPILER%"=="vs10" (call %DEVSETENV%)
@if "%THIRDPARTYCOMPILER%"=="vs10" (@msbuild CLAPACK.sln /t:ALL_BUILD /p:Configuration=Debug >msbuildDeb.log 2>msbuildDeb2.log)

::@for /r .\ %%f in (*.dll) do @copy "%%f" ..\..\..\bin\%THIRDPARTYPREFIX%\
@for /r .\ %%f in (*.lib) do @copy "%%f" ..\..\..\lib\%THIRDPARTYPREFIX%\

@echo :::::: RELEASE BUILD ::::::
@mkdir ..\release
@cd ..\release

@cmake -G "%THIRDPARTYCMAKESTR%" ../../../

@if not "%THIRDPARTYCOMPILER%"=="vs10" (@"%DEVENV%" CLAPACK.sln /build Release /project ALL_BUILD /projectconfig Release)
@if "%THIRDPARTYCOMPILER%"=="vs10" (@msbuild CLAPACK.sln /t:ALL_BUILD /p:Configuration=Release >msbuildRel.log 2>msbuildRel2.log)

::@for /r ..\%THIRDPARTYPREFIX%\ %%f in (*.dll) do @copy "%%f" ..\..\..\bin\%THIRDPARTYPREFIX%
@for /r .\ %%f in (*.lib) do @copy "%%f" ..\..\..\lib\%THIRDPARTYPREFIX%

:exit

@popd

