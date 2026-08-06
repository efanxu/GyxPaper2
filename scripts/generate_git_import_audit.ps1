param([string]$Root = 'D:\PaperProject\GyxPaper2')
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root).TrimEnd('\')
$skip = @('GIT_IMPORT_AUDIT.md','git_import_inventory.csv','git_import_excluded_paths.txt','git_import_lfs_candidates.txt')
function Rel([string]$p) { $p.Substring($Root.Length + 1).Replace('\','/') }
function Classify([string]$r,[int64]$n) {
  $l=$r.ToLowerInvariant(); $e=[IO.Path]::GetExtension($r).ToLowerInvariant()
  if($l -match '(^|/)(data|downloads|cache|caches|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.venv|venv|env|wandb|tensorboard)(/|$)'){return 'dataset/cache/environment'}
  if($l -match '(^|/)(logs?|tensorboard|wandb)(/|$)' -or $e -in @('.log','.out')){return 'logs'}
  if($e -in @('.pt','.pth','.ckpt','.bin')){return 'checkpoint'}
  if($e -in @('.npy','.npz','.h5','.hdf5')){return 'array-artifact'}
  if($e -in @('.zip','.7z','.rar','.tar','.whl','.msi','.exe','.pkg')){return 'archive/installer'}
  if($e -in @('.pyc','.pyo')){return 'bytecode'}
  if($l -match '(^|/)(results?|reports?)(/|$)'){return 'experiment-result'}
  if($e -in @('.py','.ps1','.sh','.bat','.cmd')){return 'source/script'}
  if($e -in @('.md','.txt','.rst')){return 'documentation'}
  if($e -in @('.yml','.yaml','.json','.toml','.ini','.cfg','.xml')){return 'configuration/metadata'}
  if($e -in @('.csv','.tsv')){return 'tabular-data/metadata'}
  if($e -in @('.png','.jpg','.jpeg','.svg','.pdf')){return 'figure/document'}
  return 'other'
}
function Decide([string]$r,[string]$c,[int64]$n,[string]$s) {
  $l=$r.ToLowerInvariant(); $e=[IO.Path]::GetExtension($r).ToLowerInvariant()
  if($s -eq 'Yes'){return @('No','No','possible credential or secret pattern; manual review required')}
  if($l -match '(^|/)(dataset|data)/.*\.(parquet|feather|pkl|pickle|h5|hdf5|npy|npz)$'){return @('No','No','raw or processed dataset body; retained locally')}
  if($l -match '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.ipynb_checkpoints|\.idea|\.vscode)/'){return @('No','No','local IDE or cache state')}
  if($c -in @('dataset/cache/environment','logs','checkpoint','array-artifact','bytecode')){return @('No','No','large or regenerable local artifact; retained locally')}
  if($c -eq 'archive/installer'){return @('No','No','duplicate archive or installer; source/config uploaded instead')}
  if($c -eq 'experiment-result' -and $n -gt 5MB){return @('No','No','large regenerable experiment result; summary files remain eligible')}
  if($n -ge 100MB){return @('No','No','>=100 MiB; no approved canonical LFS artifact')}
  if($n -gt 50MB){return @('No','No','>50 MiB; no approved canonical LFS artifact')}
  return @('Yes','No','source, configuration, documentation, test, metadata, or small reproducibility artifact')
}
$stack=[Collections.Generic.Stack[string]]::new(); $stack.Push($Root)
$rows=[Collections.Generic.List[object]]::new(); $reparse=[Collections.Generic.List[string]]::new()
while($stack.Count -gt 0){
  $d=$stack.Pop(); try{$children=[IO.Directory]::EnumerateFileSystemEntries($d)}catch{continue}
  foreach($p in $children){
    try{$i=Get-Item -LiteralPath $p -Force -ErrorAction Stop}catch{continue}
    if(($i.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){$reparse.Add((Rel $p));continue}
    if($i.PSIsContainer){if($i.Name -ne '.git'){$stack.Push($p)};continue}
    $r=Rel $p; if($skip -contains $r){continue}; $c=Classify $r ([int64]$i.Length); $s='No'
    if($r -match '(^|/)(\.env|.*(secret|credential|password|private[_-]?key).*)$' -or $r -match '\.(pem|key|p12|pfx)$'){$s='Yes'}
    $e=[IO.Path]::GetExtension($r).ToLowerInvariant()
    if($s -eq 'No' -and $i.Length -le 5MB -and $e -in @('.py','.ps1','.sh','.bat','.cmd','.md','.txt','.json','.yaml','.yml','.toml','.ini','.cfg','.xml','.csv')){
      try{$t=[IO.File]::ReadAllText($p);if($t -match '(?im)(api[_-]?key|access[_-]?token|password|passwd|secret|aws_secret_access_key)\s*[:=]\s*["'']?[A-Za-z0-9/+=._-]{16,}'){$s='Yes'};if($t -match '(?im)-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'){$s='Yes'}}catch{}
    }
    $dcs=Decide $r $c ([int64]$i.Length) $s; $sha=''
    if($i.Length -ge 10MB){try{$sha=(Get-Item -LiteralPath $p -Algorithm content record).record}catch{$sha='record_ERROR'}}
    $rows.Add([pscustomobject]@{RelativePath=$r;SizeBytes=[int64]$i.Length;Extension=$e;LastWriteTime=$i.LastWriteTime.ToString('o');Classification=$c;PlannedToCommit=$dcs[0];PlannedLFS=$dcs[1];ExclusionReason=$dcs[2];content record=$sha;SensitiveReview=$s})
  }
}
$rows=@($rows|Sort-Object RelativePath)
$rows|Export-Csv -LiteralPath (Join-Path $Root 'git_import_inventory.csv') -NoTypeInformation -Encoding UTF8
$ex=@('# path<TAB>size_bytes<TAB>content record<TAB>classification<TAB>reason')+@($rows|? PlannedToCommit -eq 'No'|%{'{0}`t{1}`t{2}`t{3}`t{4}'-f $_.RelativePath,$_.SizeBytes,$_.content record,$_.Classification,$_.ExclusionReason})
$ex|Set-Content -LiteralPath (Join-Path $Root 'git_import_excluded_paths.txt') -Encoding UTF8
$lf=@('# No files approved for Git LFS in this import; no canonical checkpoint was selected.')
$lf|Set-Content -LiteralPath (Join-Path $Root 'git_import_lfs_candidates.txt') -Encoding UTF8
$tot=($rows|measure SizeBytes -Sum).Sum; $pl=@($rows|? PlannedToCommit -eq 'Yes'); $no=@($rows|? PlannedToCommit -eq 'No'); $large=@($rows|? SizeBytes -gt 10MB|sort SizeBytes -Descending)
$top=@($rows|%{[pscustomobject]@{Top=(($_.RelativePath-split '/',2)[0]);Bytes=$_.SizeBytes}}|group Top|%{$b=($_.Group|measure Bytes -Sum).Sum;[pscustomobject]@{Top=$_.Name;Files=$_.Count;GiB=[math]::Round($b/1GB,2)}}|sort GiB -Descending)
$out=[Collections.Generic.List[string]]::new();$out.Add('# Git import audit');$out.Add('');$out.Add("Generated: $(Get-Date -Format o)");$out.Add('');$out.Add('## Summary');$out.Add('');$out.Add("- Files scanned: $($rows.Count)");$out.Add("- Total size: $tot bytes ($([math]::Round($tot/1GB,2)) GiB)");$out.Add("- Planned for commit: $($pl.Count) files, $((($pl|measure SizeBytes -Sum).Sum)) bytes");$out.Add("- Excluded but retained locally: $($no.Count) files, $((($no|measure SizeBytes -Sum).Sum)) bytes");$out.Add("- >10 MiB: $(@($rows|? SizeBytes -gt 10MB).Count); >50 MiB: $(@($rows|? SizeBytes -gt 50MB).Count); >=100 MiB: $(@($rows|? SizeBytes -ge 100MB).Count)");$out.Add("- Reparse points skipped: $($reparse.Count)");$out.Add('');$out.Add('## Policy');$out.Add('');$out.Add('- Source, scripts, configuration, documentation, tests, schemas, location metadata, and small summaries are eligible for normal Git.');$out.Add('- Dataset bodies, checkpoints, arrays, logs, caches, bytecode, environments, installers, duplicate archives, and large regenerable results are excluded and remain on disk.');$out.Add('- No file was approved for Git LFS; no canonical checkpoint was selected.');$out.Add('- No local data, checkpoint, result, or history file was deleted.');$out.Add('');$out.Add('## Top-level inventory');$out.Add('');$out.Add('| Top-level path | Files | GiB |');$out.Add('|---|---:|---:|');foreach($x in $top){$out.Add("| $($x.Top) | $($x.Files) | $($x.GiB) |")};$out.Add('');$out.Add('## Large files');$out.Add('');$out.Add('All files above 10 MiB are recorded in `git_import_inventory.csv`; excluded files are listed with content record in `git_import_excluded_paths.txt`.');$out.Add('');$out.Add('| Size MiB | Relative path | Classification | Plan | Reason |');$out.Add('|---:|---|---|---|---|');foreach($x in $large){$out.Add("| $([math]::Round($x.SizeBytes/1MB,1)) | ``$($x.RelativePath)`` | $($x.Classification) | $($x.PlannedToCommit) | $($x.ExclusionReason) |")};$out.Add('');$out.Add('## Sensitive-information review');$out.Add('');$ss=@($rows|? SensitiveReview -eq 'Yes');if($ss.Count -eq 0){$out.Add('- No filename or concrete secret-assignment pattern was detected by the non-secret scan.')}else{foreach($x in $ss){$out.Add("- Review required: ``$($x.RelativePath)``; path only, excluded from commit.")}};[IO.File]::WriteAllLines((Join-Path $Root 'GIT_IMPORT_AUDIT.md'),$out,[Text.UTF8Encoding]::new($false))
Write-Output "Generated audit for $($rows.Count) files; planned=$($pl.Count); excluded=$($no.Count); sensitive_review=$($ss.Count)"
