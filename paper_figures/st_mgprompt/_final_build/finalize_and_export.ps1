$ErrorActionPreference = 'Stop'

$pptPath = 'D:\PaperProject\GyxPaper2\paper_figures\st_mgprompt\ST-MGPrompt_Final_Figures.pptx'
$outDir = 'D:\PaperProject\GyxPaper2\paper_figures\st_mgprompt'

$pngNames = @(
    'Fig1_STMGPrompt_Overall_Framework_Final.png',
    'Fig2_Dual_Semantic_Graph_Construction_Final.png',
    'Fig3_MacroPrompt_Bidirectional_CrossFusion_Final.png'
)
$pdfNames = @(
    'Fig1_STMGPrompt_Overall_Framework_Final.pdf',
    'Fig2_Dual_Semantic_Graph_Construction_Final.pdf',
    'Fig3_MacroPrompt_Bidirectional_CrossFusion_Final.pdf'
)

# Artifact Tool exports editable shapes with an Open XML noGrp lock. Remove only
# that lock so PowerPoint can create the requested Selection Pane groups.
Add-Type -AssemblyName System.IO.Compression
$fileStream = [System.IO.File]::Open($pptPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$archive = [System.IO.Compression.ZipArchive]::new($fileStream, [System.IO.Compression.ZipArchiveMode]::Update, $false)
$slideEntryNames = @($archive.Entries | Where-Object { $_.FullName -like 'ppt/slides/slide*.xml' } | ForEach-Object { $_.FullName })
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($entryName in $slideEntryNames) {
    $entry = $archive.GetEntry($entryName)
    $reader = [System.IO.StreamReader]::new($entry.Open(), $utf8NoBom)
    $xml = $reader.ReadToEnd()
    $reader.Close()
    $updated = $xml.Replace(' noGrp="1"', '')
    if ($updated -ne $xml) {
        $entry.Delete()
        $newEntry = $archive.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
        $writer = [System.IO.StreamWriter]::new($newEntry.Open(), $utf8NoBom)
        $writer.Write($updated)
        $writer.Close()
    }
}
$archive.Dispose()
$fileStream.Dispose()

function Set-TimesNewRoman([object]$shape) {
    if ($shape.Type -eq 6) {
        for ($j = 1; $j -le $shape.GroupItems.Count; $j++) {
            Set-TimesNewRoman $shape.GroupItems.Item($j)
        }
    }
    if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
        $textRange = $shape.TextFrame.TextRange
        $textRange.Font.Name = 'Times New Roman'
        try { $textRange.Font.NameFarEast = 'Times New Roman' } catch {}
        if ($shape.Name -eq 'FIGURE_TITLE') {
            try { $shape.TextFrame.AutoSize = 0 } catch {}
            try {
                $shape.TextFrame.MarginLeft = 0
                $shape.TextFrame.MarginRight = 0
                $shape.TextFrame.MarginTop = 0
                $shape.TextFrame.MarginBottom = 0
            } catch {}
            $textRange.Font.Size = 28
            $textRange.Font.Bold = -1
        } elseif ($shape.Name -match '(__TITLE|__BAND_TITLE)$') {
            try { $shape.TextFrame.AutoSize = 0 } catch {}
            try {
                $shape.TextFrame.MarginLeft = 0
                $shape.TextFrame.MarginRight = 0
                $shape.TextFrame.MarginTop = 0
                $shape.TextFrame.MarginBottom = 0
            } catch {}
            $textRange.Font.Size = 18
            $textRange.Font.Bold = -1
        } elseif ([double]$textRange.Font.Size -lt 10) {
            try { $shape.TextFrame.AutoSize = 0 } catch {}
            try {
                $shape.TextFrame.MarginLeft = 1
                $shape.TextFrame.MarginRight = 1
                $shape.TextFrame.MarginTop = 0.5
                $shape.TextFrame.MarginBottom = 0.5
            } catch {}
            $textRange.Font.Size = 10
            $textRange.Font.Bold = 0
        }
    }
}

function Group-ByPrefix([object]$slide, [string]$prefix, [string]$groupName) {
    $names = New-Object System.Collections.Generic.List[object]
    for ($i = 1; $i -le $slide.Shapes.Count; $i++) {
        $shapeName = $slide.Shapes.Item($i).Name
        if ($shapeName.StartsWith($prefix, [System.StringComparison]::Ordinal)) {
            $names.Add($shapeName)
        }
    }
    if ($names.Count -gt 1) {
        $range = $slide.Shapes.Range([object[]]$names.ToArray())
        $group = $range.Group()
        $group.Name = $groupName
    }
}

$app = New-Object -ComObject PowerPoint.Application
$app.DisplayAlerts = 1
$app.Visible = -1
$pres = $app.Presentations.Open($pptPath, $false, $false, $true)

foreach ($slide in $pres.Slides) {
    for ($i = 1; $i -le $slide.Shapes.Count; $i++) {
        Set-TimesNewRoman $slide.Shapes.Item($i)
    }
}

Group-ByPrefix $pres.Slides.Item(1) 'FIG1_PANEL_A__' 'FIG1_PANEL_A'
Group-ByPrefix $pres.Slides.Item(1) 'FIG1_PANEL_B__' 'FIG1_PANEL_B'
Group-ByPrefix $pres.Slides.Item(1) 'FIG1_PANEL_C__' 'FIG1_PANEL_C'
Group-ByPrefix $pres.Slides.Item(1) 'FIG1_PANEL_D__' 'FIG1_PANEL_D'
Group-ByPrefix $pres.Slides.Item(2) 'FIG2_DISTANCE__' 'FIG2_DISTANCE'
Group-ByPrefix $pres.Slides.Item(2) 'FIG2_PRIOR__' 'FIG2_PRIOR'
Group-ByPrefix $pres.Slides.Item(2) 'FIG2_ADAPTIVE__' 'FIG2_ADAPTIVE'
Group-ByPrefix $pres.Slides.Item(3) 'FIG3_PROMPT__' 'FIG3_PROMPT'
Group-ByPrefix $pres.Slides.Item(3) 'FIG3_CROSSFUSION__' 'FIG3_CROSSFUSION'
Group-ByPrefix $pres.Slides.Item(3) 'FIG3_OUTPUT__' 'FIG3_OUTPUT'

$pres.Save()

for ($i = 1; $i -le 3; $i++) {
    $pngPath = Join-Path $outDir $pngNames[$i - 1]
    $pres.Slides.Item($i).Export($pngPath, 'PNG', 3840, 2160)

    $pdfPath = Join-Path $outDir $pdfNames[$i - 1]
    $printRange = $pres.PrintOptions.Ranges.Add($i, $i)
    $pres.ExportAsFixedFormat(
        $pdfPath,
        2,
        1,
        0,
        1,
        1,
        0,
        $printRange,
        4,
        '',
        -1,
        -1,
        -1,
        -1,
        0
    )
    $pres.PrintOptions.Ranges.ClearAll()
}

$pres.Close()
try { $app.Quit() } catch {}

Get-Item -LiteralPath $pptPath | Select-Object FullName,Length,LastWriteTime
foreach ($name in $pngNames + $pdfNames) {
    Get-Item -LiteralPath (Join-Path $outDir $name) | Select-Object FullName,Length,LastWriteTime
}
