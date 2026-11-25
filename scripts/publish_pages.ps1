$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (git status --porcelain) {
    throw 'Commit the reviewed source changes before publishing.'
}
$pageCommit = git subtree split --prefix=web HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not extract the static site.' }
git push origin "${pageCommit}:refs/heads/gh-pages"
if ($LASTEXITCODE -ne 0) { throw 'Static branch push failed; do not force-push.' }
Write-Output 'Static branch pushed. Check the GitHub Pages deployment and public URL.'
