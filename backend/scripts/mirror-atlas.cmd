@echo off
rem Refreshes the local MongoDB mirror from Atlas; run by the "Kenya Watch Atlas mirror" scheduled task.
cd /d "%~dp0.."
node scripts\mirror-atlas.js >> mirror.log 2>&1
