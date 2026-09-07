@echo off
setlocal

set "PY311=C:\Users\ashsh\anaconda3\envs\autogen311\python.exe"
set "TARGET_PER_EMOTION=100"
set "REVIEW_SAMPLE_PER_CLASS=50"
set "MAX_REVIEW_TOTAL=500"
set "CLUSTER_CLEAN_FEATURE=multi_feature"

if not exist "%PY311%" (
  echo Python not found: %PY311%
  echo.
  echo Press any key to close...
  pause >nul
  exit /b 1
)

echo.
echo Emotion workflow limits
echo Press Enter to use the default shown in brackets.
set /p "TARGET_INPUT=Target clean samples per emotion [%TARGET_PER_EMOTION%]: "
if not "%TARGET_INPUT%"=="" set "TARGET_PER_EMOTION=%TARGET_INPUT%"
set /p "REVIEW_INPUT=Review shortlist per emotion [%REVIEW_SAMPLE_PER_CLASS%]: "
if not "%REVIEW_INPUT%"=="" set "REVIEW_SAMPLE_PER_CLASS=%REVIEW_INPUT%"
set /p "MAX_REVIEW_INPUT=Maximum total review shortlist [%MAX_REVIEW_TOTAL%]: "
if not "%MAX_REVIEW_INPUT%"=="" set "MAX_REVIEW_TOTAL=%MAX_REVIEW_INPUT%"
set /p "FEATURE_INPUT=Cluster-clean feature multi_feature/mfcc_separable/mfcc/mfcc_rich/acoustic [%CLUSTER_CLEAN_FEATURE%]: "
if not "%FEATURE_INPUT%"=="" set "CLUSTER_CLEAN_FEATURE=%FEATURE_INPUT%"

set "COMMON_ARGS=--target-per-emotion %TARGET_PER_EMOTION% --review-sample-per-class %REVIEW_SAMPLE_PER_CLASS% --max-review-total %MAX_REVIEW_TOTAL% --cluster-clean-feature %CLUSTER_CLEAN_FEATURE%"
echo.
echo Using: %COMMON_ARGS%
if /I "%CLUSTER_CLEAN_FEATURE%"=="acoustic" (
  echo Clean target output folder will use: cluster_clean_target_%TARGET_PER_EMOTION%
) else (
  echo Clean target output folder will use: cluster_clean_target_%TARGET_PER_EMOTION%_%CLUSTER_CLEAN_FEATURE%
)
echo Balanced review output folder will use: balanced_review_target_%TARGET_PER_EMOTION%
echo.

"%PY311%" "%~dp0emotion_workflow_runner.py" %COMMON_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Press any key to close...
pause >nul
endlocal
exit /b %EXIT_CODE%
