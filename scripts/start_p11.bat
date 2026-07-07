@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  Zeus P11-V4 — lancement du bot (paper par defaut)
REM  Prerequis : terminal MT5 ouvert et connecte, variables ZEUS_MT5_*
REM  definies (setx), Python 3.11+ dans le PATH.
REM ═══════════════════════════════════════════════════════════════════
cd /d "%~dp0.."

if "%ZEUS_MT5_LOGIN%"=="" (
    echo ERREUR : variable ZEUS_MT5_LOGIN non definie.
    echo Lance :  setx ZEUS_MT5_LOGIN "12345678"  ^(puis rouvre ce script^)
    pause
    exit /b 1
)

echo ─────────────────────────────────────────────
echo  Zeus P11-V4 — demarrage en mode PAPER
echo  Config : config\p11_v4.yaml
echo  Arret  : Ctrl+C
echo ─────────────────────────────────────────────
python -m zeus.live.run_p11 %*

pause
