 ..\..\..\crf_test -m model_mine test_data_mine.txt -o result.txt

@echo off
setlocal enabledelayedexpansion

set "file=result.txt"

if not exist "!file!" (
    echo 文件不存在: "!file!"
    exit /b 1
)

(for /f "delims=" %%A in (!file!) do (
    set "line=%%A"
    set "line=!line:_=-!"
    echo !line!
)) > tmpfile.txt

move /y tmpfile.txt "!file!"

echo 替换完成。

endlocal

perl conlleval.pl -d "\t" <result.txt >result.info
