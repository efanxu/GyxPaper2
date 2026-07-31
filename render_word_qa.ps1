$inputPath = (Resolve-Path "D:\PaperProject\GyxPaper2\ST-MGPrompt_revised_manuscript.docx").Path
$outputDir = "D:\PaperProject\GyxPaper2\qa_render"
$pdfPath = Join-Path $outputDir "ST-MGPrompt_revised_manuscript.pdf"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
try {
    Write-Output "Opening Word"
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    Write-Output "Opening document"
    $doc = $word.Documents.Open($inputPath, $false, $true)
    Write-Output "Exporting PDF"
    $doc.SaveAs2($pdfPath, 17)
    Write-Output "Closing Word"
    $doc.Close($false)
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    Write-Output $pdfPath
} catch {
    Write-Error $_
    if ($doc) { $doc.Close($false) }
    if ($word) { $word.Quit() }
    exit 1
}
