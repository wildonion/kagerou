@echo off
REM ============================================================
REM  Kagerou - GPU Video Transcoding SDK
REM  Single executable: kagerou.exe
REM
REM  Usage:
REM    build.bat                      auto-detect GPU + stub mode
REM    build.bat sdk                  build with Video Codec SDK
REM    build.bat sm_89                explicit arch
REM    build.bat test                 build and run tests
REM    build.bat all                  build kagerou.exe + tests + virtualcam
REM    build.bat all sdk              build everything with SDK
REM    build.bat virtualcam sdk minimp4 onnx
REM                                   build virtual camera EXE + DirectShow DLL
REM                                   and register it (UAC prompt appears)
REM    build.bat vcam_dll             build + register only the virtualcam DLL
REM    build.bat vcam_test            verify virtualcam enumerates + streams
REM    build.bat ai_diag <frame.raw>  probe AI model I/O with a real frame
REM
REM  SDK Integration:
REM    1. Download Video Codec SDK from nvidia.com
REM    2. Place headers in sdk/nvidia_video_codec_sdk/Interface/
REM    3. Run: build.bat sdk
REM
REM  Output: bin\kagerou.exe, bin\kagerou_test.exe
REM ============================================================
setlocal EnableDelayedExpansion

REM Clear poisoned CL env var (MSVC appends %CL% to every cl.exe command line;
REM a stale machine-level CL e.g. pointing at an uninstalled MSVC version
REM breaks all compilation with D9024/D9027 + silent nvcc exit 2).
set CL=

REM ---------- parse args ----------
set TARGET=%1
set USE_SDK=
set USE_MINIMP4=
set USE_ONNX=
set SM_ARGS=
if "%TARGET%"=="sdk" set USE_SDK=1 & set TARGET=all
if "%TARGET%"=="minimp4" set USE_MINIMP4=1 & set TARGET=all
if "%TARGET%"=="mp4" set USE_MINIMP4=1 & set TARGET=all
if "%TARGET%"=="onnx" set USE_ONNX=1 & set TARGET=all
if "%2%"=="sdk" set USE_SDK=1
if "%2%"=="minimp4" set USE_MINIMP4=1
if "%2%"=="mp4" set USE_MINIMP4=1
if "%2%"=="onnx" set USE_ONNX=1
if "%3%"=="sdk" set USE_SDK=1
if "%3%"=="minimp4" set USE_MINIMP4=1
if "%3%"=="mp4" set USE_MINIMP4=1
if "%3%"=="onnx" set USE_ONNX=1
if "%4%"=="sdk" set USE_SDK=1
if "%4%"=="minimp4" set USE_MINIMP4=1
if "%4%"=="mp4" set USE_MINIMP4=1
if "%4%"=="onnx" set USE_ONNX=1
if "%TARGET%"=="sm_86"  set SM_ARGS=sm_86
if "%TARGET%"=="sm_89"  set SM_ARGS=sm_89
if "%TARGET%"=="sm_120" set SM_ARGS=sm_120
if "%TARGET%"=="all"    set TARGET=all
if "%TARGET%"=="test"   set TARGET=test
if "%TARGET%"=="" set TARGET=all

REM ---------- locate CUDA 12.x toolkit (newest first) ----------
set "CUDA_TOOLKIT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
set "CUDA_VER="
for %%v in (v13.3 v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
  if not defined CUDA_VER if exist "%CUDA_TOOLKIT%\%%v\bin\nvcc.exe" set "CUDA_VER=%%v"
)
if not defined CUDA_VER (echo ERROR: no CUDA 12.x found under %CUDA_TOOLKIT% & exit /b 1)
set "NVCC=%CUDA_TOOLKIT%\%CUDA_VER%\bin\nvcc.exe"
echo Using CUDA %CUDA_VER% : %NVCC%

REM ---------- locate MSVC via vswhere ----------
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set cl=
setlocal EnableDelayedExpansion

REM ---------- output dir ----------
set "OUTDIR=bin"
if not exist %OUTDIR% mkdir %OUTDIR%

REM ---------- SDK detection ----------
set "SDK_INC=sdk\nvidia_video_codec_sdk\Interface"
set "SDK_DEFINES=-DKAGEROU_NO_VCODEC_SDK"
set "SDK_LIBS="
set "SDK_INCLUDE="
if "!USE_SDK!"=="1" if exist "!SDK_INC!\nvEncodeAPI.h" set "SDK_DEFINES=-DKAGEROU_USE_NVDEC_SDK -DKAGEROU_USE_NVENC_SDK" & set "SDK_LIBS=-Lsdk\nvidia_video_codec_sdk\Lib\Win64 -lnvcuvid -lnvEncodeAPI -lcuda" & set "SDK_INCLUDE=-I!SDK_INC!" & echo [SDK] Found NVIDIA Video Codec SDK headers
if "!USE_SDK!"=="1" if exist "!SDK_INC!\nvEncodeAPI.h" if "!USE_MINIMP4!"=="1" set "SDK_DEFINES=-DKAGEROU_USE_NVDEC_SDK -DKAGEROU_USE_NVENC_SDK -DKAGEROU_USE_MINIMP4" & echo [SDK] Also enabling MP4/MKV container support (minimp4)
if "!USE_SDK!"=="1" if not exist "!SDK_INC!\nvEncodeAPI.h" echo [SDK] WARNING: Video Codec SDK headers not found at !SDK_INC! & echo [SDK] Download the NVENC NVDEC SDK from nvidia.com & echo [SDK] Falling back to stub mode
if not "!USE_SDK!"=="1" if "!USE_MINIMP4!"=="1" set "SDK_DEFINES=-DKAGEROU_NO_VCODEC_SDK -DKAGEROU_USE_MINIMP4" & echo [SDK] Building with MP4/MKV container support (minimp4)
if not "!USE_SDK!"=="1" if not "!USE_MINIMP4!"=="1" echo [SDK] Building in stub mode (no NVDEC/NVENC hardware encode/decode) & echo [SDK] Use: build.bat sdk   for full hardware acceleration & echo [SDK] Use: build.bat minimp4   for MP4/MKV container support

REM ---------- ONNX Runtime detection ----------
set "ONNX_INC=sdk\onnxruntime\include"
set "ONNX_LIBS="
set "ONNX_DEFINES="
set "ONNX_LINK="
if "!USE_ONNX!"=="1" if exist "!ONNX_INC!\onnxruntime_cxx_api.h" (
    set "ONNX_DEFINES=-DKAGEROU_USE_ONNX"
    set "ONNX_INCLUDE=-I!ONNX_INC!"
    set "ONNX_LINK=-Lsdk\onnxruntime\lib -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_tensorrt"
    echo [ONNX] Found ONNX Runtime GPU headers and libs
) else if "!USE_ONNX!"=="1" (
    echo [ONNX] WARNING: ONNX Runtime not found at sdk\onnxruntime\
    echo [ONNX] Run: cd sdk\onnxruntime ^& powershell -ExecutionPolicy Bypass -File download.ps1
    echo [ONNX] Falling back to CPU-only mode
)

REM ---------- common flags ----------
set "COMMON_FLAGS=-O3 -rdc=true -use_fast_math --ptxas-options=-O3 -std=c++17 -Xcompiler /std:c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -I include -I src -lcudadevrt -cudart=static !SDK_DEFINES! !SDK_INCLUDE! !SDK_LIBS! !ONNX_DEFINES! !ONNX_INCLUDE! !ONNX_LINK!"

REM ---------- arch selection ----------
set "ARCH_FLAG=-arch=native"
if not "%SM_ARGS%"=="" (
  set "ARCH_FLAG="
  for %%a in (%SM_ARGS%) do (
    for /f "tokens=1,2 delims=_" %%x in ("%%a") do set "COMPUTE=compute_%%y"
    set "ARCH_FLAG=!ARCH_FLAG! -gencode arch=!COMPUTE!,code=%%a -gencode arch=!COMPUTE!,code=!COMPUTE!"
  )
  echo arch      : %SM_ARGS% (native SASS + embedded PTX)
) else (
  echo arch      : auto (native SASS for local GPU)
)

REM ---------- CUDA lib path ----------
set "CUDA_LIB=%CUDA_TOOLKIT%\%CUDA_VER%\lib\x64"

REM ---------- build targets ----------
if "%TARGET%"=="all" goto :build_all
if "%TARGET%"=="test" goto :build_test
if "%TARGET%"=="virtualcam" goto :build_virtualcam
if "%TARGET%"=="vcam_dll" goto :build_vcam_dll
if "%TARGET%"=="vcam_test" goto :build_vcam_test
if "%TARGET%"=="gesture_test" goto :build_gesture_test
if "%TARGET%"=="pip_test" goto :build_pip_test
if "%TARGET%"=="dxgi_test" goto :build_dxgi_test
if "%TARGET%"=="ai_diag" goto :build_ai_diag

if "%TARGET%"=="decode_order" goto :build_decode_order
if "%TARGET%"=="decode_compare" goto :build_decode_compare
echo Unknown target: %TARGET%
exit /b 1

:build_all
echo.
echo --- Building kagerou.exe ---
"%NVCC%" %COMMON_FLAGS% %ARCH_FLAG% -L"%CUDA_LIB%" -o %OUTDIR%\kagerou.exe examples\00_kagerou.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\kagerou.exe
call :build_test
call :build_virtualcam
echo.
echo === All builds complete ===
goto :done

:build_test
echo.
echo --- Building + running tests ---
"%NVCC%" %COMMON_FLAGS% %ARCH_FLAG% -L"%CUDA_LIB%" -o %OUTDIR%\kagerou_test.exe tests\test_filters.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\kagerou_test.exe
echo Running tests...
%OUTDIR%\kagerou_test.exe
if errorlevel 1 (echo TESTS FAILED & exit /b 1)
echo All tests passed.
goto :eof


:done
endlocal
exit /b 0

:build_decode_order
echo.
echo --- Building decode_order test ---
"%NVCC%" %COMMON_FLAGS% %ARCH_FLAG% -L"%CUDA_LIB%" -o %OUTDIR%\kagerou_decode_order.exe tests\test_decode_order.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\kagerou_decode_order.exe
goto :eof

:build_decode_compare
echo.
echo --- Building decode_compare test ---
"%NVCC%" %COMMON_FLAGS% %ARCH_FLAG% -L"%CUDA_LIB%" -o %OUTDIR%\kagerou_decode_compare.exe tests\test_decode_compare.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\kagerou_decode_compare.exe
goto :eof

:build_virtualcam
echo.
echo --- Building kagerou_virtualcam.exe (virtual camera) ---
"%NVCC%" %COMMON_FLAGS% %ARCH_FLAG% -L"%CUDA_LIB%" -o %OUTDIR%\kagerou_virtualcam.exe examples\02_virtualcam_demo.cu -luser32 -lgdi32 -lopengl32 -lcomctl32 -lcomdlg32 -lstrmiids -lole32 -loleaut32 -lmmdevapi -ld3d11 -ldxgi -lwinmm
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\kagerou_virtualcam.exe
call :build_screen_cap
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
call :build_vcam_dll
goto :eof

:build_vcam_dll
echo.
echo --- Building kagerou_virtualcam.dll (DirectShow virtual camera) ---
REM Use MSVC cl.exe for the COM DLL (NVCC not needed for pure C++)
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set "VCAM_SRC=virtualcam\kagerou_vcam_dll.cpp virtualcam\kagerou_vcam_filter.cpp"
set "VCAM_INC=-I include -I src"
set "VCAM_OUT=%OUTDIR%\kagerou_virtualcam.dll"
set cl=
cl /LD /O2 /EHsc /Fo"%OUTDIR%\obj\\" %VCAM_SRC% %VCAM_INC% /link /DLL /OUT:%VCAM_OUT% /IMPLIB:"%OUTDIR%\obj\vcam.lib" /DEF:virtualcam\kagerou_vcam.def strmiids.lib ole32.lib oleaut32.lib advapi32.lib user32.lib
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %VCAM_OUT%
echo.
echo --- Registering virtual camera (UAC prompt appears, needed for Chrome) ---
powershell -Command "Start-Process regsvr32 -ArgumentList '/s \"%CD%\%VCAM_OUT%\"' -Verb RunAs -Wait"
echo Registered. Restart camera apps to pick up "Kagerou Virtual Camera".
goto :eof

:build_vcam_test
echo.
echo --- Building + running virtualcam device test ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set cl=
cl /nologo /EHsc /Fo"%OUTDIR%\obj\\" tests\test_vcam_enum.cpp /Fe:%OUTDIR%\test_vcam_enum.exe /link strmiids.lib ole32.lib oleaut32.lib
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\test_vcam_enum.exe
%OUTDIR%\test_vcam_enum.exe
if errorlevel 1 (echo VCAM TEST FAILED & exit /b 1)
echo VCAM TEST PASSED.
goto :eof

:build_gesture_test
echo.
echo --- Building + running gesture classifier test (no GPU needed) ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set cl=
cl /nologo /EHsc /Fo"%OUTDIR%\obj\\" tests\test_gesture.cpp /Fe:%OUTDIR%\test_gesture.exe /I include /I "%CUDA_TOOLKIT%\%CUDA_VER%\include"
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\test_gesture.exe
%OUTDIR%\test_gesture.exe
if errorlevel 1 (echo GESTURE TEST FAILED & exit /b 1)
echo GESTURE TEST PASSED.
goto :eof

:build_screen_cap
echo.
echo --- Building screen capture helper (no CUDA) ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set cl=
cl /nologo /EHsc /Fo"%OUTDIR%\obj\\" screen_cap\screen_cap.cpp /Fe:%OUTDIR%\screen_cap.exe /I include /link d3d11.lib dxgi.lib
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\screen_cap.exe
goto :eof

:build_pip_test
echo.
echo --- Building + running PiP compositor test (no GPU needed) ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set cl=
cl /nologo /EHsc /Fo"%OUTDIR%\obj\\" tests\test_pip.cpp /Fe:%OUTDIR%\test_pip.exe /I include
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\test_pip.exe
%OUTDIR%\test_pip.exe
if errorlevel 1 (echo PIP TEST FAILED & exit /b 1)
echo PIP TEST PASSED.
goto :eof

:build_dxgi_test
echo.
echo --- Building DXGI desktop-duplication probe ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
set cl=
cl /nologo /EHsc /Fo"%OUTDIR%\obj\\" tests\test_dxgi.cpp /Fe:%OUTDIR%\test_dxgi.exe /link d3d11.lib dxgi.lib
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\test_dxgi.exe
%OUTDIR%\test_dxgi.exe
goto :eof

:build_ai_diag
echo.
echo --- Building AI probe (needs sdk\onnxruntime + a 640x480 rgb24 frame) ---
if not exist %OUTDIR%\obj mkdir %OUTDIR%\obj
"%NVCC%" -O2 -std=c++17 -Xcompiler /std:c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -I include -I src -DKAGEROU_USE_ONNX -I sdk\onnxruntime\include -L sdk\onnxruntime\lib -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_tensorrt -o %OUTDIR%\test_ai_diag.exe tests\test_ai_diag.cu -luser32
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: %OUTDIR%\test_ai_diag.exe
echo Run: %OUTDIR%\test_ai_diag.exe ^<frame.raw^>  (640x480 rgb24)
goto :eof
