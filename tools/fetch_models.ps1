$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$root = 'D:\zeninfer'
$mdir = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $mdir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root 'tools') | Out-Null

$base = 'https://hf-mirror.com'
$files = @(
  @{ url = "$base/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";                 name = 'tokenizer.json' },
  @{ url = "$base/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"; name = 'qwen2.5-0.5b-instruct-q8_0.gguf' },
  @{ url = "$base/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"; name = 'qwen2.5-0.5b-instruct-q4_0.gguf' },
  @{ url = "$base/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"; name = 'qwen2.5-0.5b-instruct-fp16.gguf' }
)

foreach ($f in $files) {
  $out = Join-Path $mdir $f.name
  Write-Output "=== DOWNLOAD $($f.name) ==="
  & curl.exe -L --fail --retry 3 --retry-delay 3 -C - --no-progress-meter -o $out $f.url
  if (Test-Path $out) {
    $len = (Get-Item $out).Length
    Write-Output ("DONE {0}  {1:N1} MB  exit={2}" -f $f.name, ($len / 1MB), $LASTEXITCODE)
  } else {
    Write-Output "MISSING $($f.name)"
  }
}
Write-Output "=== ALL DOWNLOADS FINISHED ==="
