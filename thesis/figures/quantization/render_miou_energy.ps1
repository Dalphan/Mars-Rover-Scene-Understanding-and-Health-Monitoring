param(
    [string]$Source = 'D:\Daniele\Scuola\Magistrale\Tesi\Codice\thesis\latex\Chapters\5 Evaluation of Lightweight Scene Understanding.tex',
    [string]$Output = (Join-Path $PSScriptRoot 'quantization_miou_energy_updated.png')
)
# Plot directly from the test-results table, preserving its 20 measurements.
# Logarithmic energy avoids compressing the other points because of FP32 DeepLabV3.
# Render to a separate preview first, so the original image is preserved for QA.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$invariant = [Globalization.CultureInfo]::InvariantCulture
$sourceText = Get-Content -LiteralPath $Source -Raw -Encoding UTF8
$labelIndex = $sourceText.IndexOf('\label{tab:s5mars-quantization-accuracy-efficiency}')
if ($labelIndex -lt 0) { throw 'Quantization table label not found' }
$tableStart = $sourceText.LastIndexOf('\begin{table}', $labelIndex)
$tableEnd = $sourceText.IndexOf('\end{table}', $labelIndex)
$tableText = $sourceText.Substring($tableStart, $tableEnd - $tableStart)
$models = [ordered]@{
    'U-Net (ResNet34)' = @('U-Net / R34', '#0072B2')
    'U-Net (MobileNetV2)' = @('U-Net / MV2', '#009E73')
    'DeepLabV3 (ResNet34)' = @('DeepLabV3 / R34', '#D55E00')
    'DeepLabV3+ (MobileNetV2)' = @('DeepLabV3+ / MV2', '#CC79A7')
    'SegFormer-B0' = @('SegFormer-B0', '#E69F00')
}
$shapes = [ordered]@{ 'FP32' = 'circle'; 'FP16' = 'square'; 'PTQ INT8' = 'triangle'; 'QAT INT8' = 'diamond' }
$points = @()
$currentModel = $null
foreach ($line in ($tableText -split '\r?\n')) {
    if ($line -match '\\textit\{([^}]+)\}') { $currentModel = $Matches[1] }
    if ($line -match '^\s*(FP32|FP16|PTQ INT8|QAT INT8)\s*&') {
        $fields = $line -split '&'
        if ($fields.Count -ne 6 -or -not $models.Contains($currentModel)) { throw "Unexpected row: $line" }
        $points += [pscustomobject]@{
            model = $currentModel
            variant = $fields[0].Trim()
            test_miou_percent = [double]::Parse($fields[1].Trim(), $invariant)
            energy_mj_per_image = [double]::Parse($fields[4].Trim(), $invariant)
        }
    }
}
if ($points.Count -ne 20) { throw "Expected 20 configurations, found $($points.Count)" }
foreach ($model in $models.Keys) {
    foreach ($variant in $shapes.Keys) {
        $matching = @($points | Where-Object { $_.model -eq $model -and $_.variant -eq $variant })
        if ($matching.Count -ne 1) { throw "Missing or duplicate point: $model / $variant" }
    }
}
$frontier = @($points | Where-Object {
    $candidate = $_
    $dominators = @($points | Where-Object {
        $_.energy_mj_per_image -le $candidate.energy_mj_per_image -and
        $_.test_miou_percent -ge $candidate.test_miou_percent -and
        ($_.energy_mj_per_image -lt $candidate.energy_mj_per_image -or $_.test_miou_percent -gt $candidate.test_miou_percent)
    })
    $dominators.Count -eq 0
})
$chartLeft = 160.0; $chartRight = 1950.0; $chartTop = 40.0; $chartBottom = 1040.0
$energyMinimum = 100.0; $energyMaximum = 4500.0; $miouMinimum = 57.5; $miouMaximum = 79.0
function Get-X([double]$Value) {
    $chartLeft + ([math]::Log10($Value) - [math]::Log10($energyMinimum)) /
        ([math]::Log10($energyMaximum) - [math]::Log10($energyMinimum)) * ($chartRight - $chartLeft)
}
function Get-Y([double]$Value) { $chartBottom - ($Value-$miouMinimum) / ($miouMaximum-$miouMinimum) * ($chartBottom-$chartTop) }
foreach ($point in $points) {
    if ($point.energy_mj_per_image -lt $energyMinimum -or $point.energy_mj_per_image -gt $energyMaximum -or
        $point.test_miou_percent -lt $miouMinimum -or $point.test_miou_percent -gt $miouMaximum) { throw 'Point outside plot limits' }
}
$bitmap = [Drawing.Bitmap]::new(2000, 1430)
$bitmap.SetResolution(240, 240)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([Drawing.Color]::White)
$tickFont = [Drawing.Font]::new('Times New Roman', 30, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$labelFont = [Drawing.Font]::new('Times New Roman', 36, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$annotationFont = [Drawing.Font]::new('Times New Roman', 30, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$legendFont = [Drawing.Font]::new('Times New Roman', 30, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$axisPen = [Drawing.Pen]::new([Drawing.Color]::Black, 3)
$gridPen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#dedede'), 2)
$edgePen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#303030'), 2.4)
$annotationPen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#555555'), 2.4)
$centerFormat = [Drawing.StringFormat]::new()
$centerFormat.Alignment = [Drawing.StringAlignment]::Center
$centerFormat.LineAlignment = [Drawing.StringAlignment]::Center
$rightFormat = [Drawing.StringFormat]::new()
$rightFormat.Alignment = [Drawing.StringAlignment]::Far
$rightFormat.LineAlignment = [Drawing.StringAlignment]::Center
function Draw-Marker([double]$X, [double]$Y, [string]$Shape, [Drawing.Brush]$Fill, [double]$Radius = 16) {
    if ($Shape -eq 'circle') {
        $graphics.FillEllipse($Fill, [single]($X-$Radius), [single]($Y-$Radius), [single](2*$Radius), [single](2*$Radius))
        $graphics.DrawEllipse($edgePen, [single]($X-$Radius), [single]($Y-$Radius), [single](2*$Radius), [single](2*$Radius))
    } elseif ($Shape -eq 'square') {
        $graphics.FillRectangle($Fill, [single]($X-$Radius), [single]($Y-$Radius), [single](2*$Radius), [single](2*$Radius))
        $graphics.DrawRectangle($edgePen, [single]($X-$Radius), [single]($Y-$Radius), [single](2*$Radius), [single](2*$Radius))
    } else {
        if ($Shape -eq 'triangle') {
            [Drawing.PointF[]]$vertices = @([Drawing.PointF]::new($X,$Y-$Radius), [Drawing.PointF]::new($X-$Radius,$Y+$Radius), [Drawing.PointF]::new($X+$Radius,$Y+$Radius))
        } elseif ($Shape -eq 'diamond') {
            $diagonal = $Radius * 1.3
            [Drawing.PointF[]]$vertices = @([Drawing.PointF]::new($X,$Y-$diagonal), [Drawing.PointF]::new($X-$diagonal,$Y), [Drawing.PointF]::new($X,$Y+$diagonal), [Drawing.PointF]::new($X+$diagonal,$Y))
        } else { throw "Unknown marker: $Shape" }
        $graphics.FillPolygon($Fill, $vertices)
        $graphics.DrawPolygon($edgePen, $vertices)
    }
}
try {
    foreach ($tick in @(100,200,400,800,1600,3200)) {
        $x = Get-X $tick
        $graphics.DrawLine($gridPen, [single]$x, [single]$chartTop, [single]$x, [single]$chartBottom)
        $graphics.DrawLine($axisPen, [single]$x, [single]$chartBottom, [single]$x, [single]($chartBottom+12))
        $graphics.DrawString([string]$tick, $tickFont, [Drawing.Brushes]::Black, [Drawing.PointF]::new($x,1080), $centerFormat)
    }
    foreach ($tick in @(60,63,66,69,72,75,78)) {
        $y = Get-Y $tick
        $graphics.DrawLine($gridPen, [single]$chartLeft, [single]$y, [single]$chartRight, [single]$y)
        $graphics.DrawLine($axisPen, [single]($chartLeft-12), [single]$y, [single]$chartLeft, [single]$y)
        $graphics.DrawString([string]$tick, $tickFont, [Drawing.Brushes]::Black, [Drawing.PointF]::new($chartLeft-24,$y), $rightFormat)
    }
    $graphics.DrawLine($axisPen, [single]$chartLeft, [single]$chartTop, [single]$chartLeft, [single]$chartBottom)
    $graphics.DrawLine($axisPen, [single]$chartLeft, [single]$chartBottom, [single]$chartRight, [single]$chartBottom)
    foreach ($point in $points) {
        $brush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml($models[$point.model][1]))
        try { Draw-Marker (Get-X $point.energy_mj_per_image) (Get-Y $point.test_miou_percent) $shapes[$point.variant] $brush }
        finally { $brush.Dispose() }
    }
    $graphics.DrawString('GPU energy per image (mJ, logarithmic scale)', $labelFont, [Drawing.Brushes]::Black, [Drawing.PointF]::new(($chartLeft+$chartRight)/2,1150), $centerFormat)
    $state = $graphics.Save()
    $graphics.TranslateTransform(45, [single](($chartTop+$chartBottom)/2))
    $graphics.RotateTransform(-90)
    $graphics.DrawString('Test mIoU (%)', $labelFont, [Drawing.Brushes]::Black, [Drawing.PointF]::new(0,0), $centerFormat)
    $graphics.Restore($state)
    $annotations = @(
        @('SegFormer-B0','QAT INT8','SegFormer QAT',470,77.5),
        @('DeepLabV3+ (MobileNetV2)','QAT INT8','DeepLabV3+ QAT',265,68.1),
        @('DeepLabV3 (ResNet34)','FP32','DeepLabV3 FP32',1700,71.1)
    )
    foreach ($annotation in $annotations) {
        $point = $points | Where-Object { $_.model -eq $annotation[0] -and $_.variant -eq $annotation[1] }
        $labelX = Get-X $annotation[3]; $labelY = Get-Y $annotation[4]
        $graphics.DrawLine($annotationPen, [single](Get-X $point.energy_mj_per_image), [single](Get-Y $point.test_miou_percent), [single]$labelX, [single]$labelY)
        $size = $graphics.MeasureString($annotation[2], $annotationFont)
        $graphics.FillRectangle([Drawing.Brushes]::White, [single]($labelX-5), [single]($labelY-$size.Height/2), [single]($size.Width+10), [single]$size.Height)
        $graphics.DrawString($annotation[2], $annotationFont, [Drawing.Brushes]::Black, [single]$labelX, [single]($labelY-$size.Height/2))
    }
    $modelEntries = @($models.GetEnumerator())
    $legendPositions = @(@(0,310,1250), @(1,310,1310), @(2,875,1250), @(3,875,1310), @(4,1460,1250))
    foreach ($position in $legendPositions) {
        $entry = $modelEntries[$position[0]]
        $brush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml($entry.Value[1]))
        try { Draw-Marker $position[1] $position[2] 'circle' $brush 13 }
        finally { $brush.Dispose() }
        $size = $graphics.MeasureString($entry.Value[0], $legendFont)
        $graphics.DrawString($entry.Value[0], $legendFont, [Drawing.Brushes]::Black, [single]($position[1]+34), [single]($position[2]-$size.Height/2))
    }
    $precisionX = @(390,730,1100,1550)
    $index = 0
    foreach ($variant in $shapes.Keys) {
        Draw-Marker $precisionX[$index] 1390 $shapes[$variant] ([Drawing.Brushes]::White) 13
        $size = $graphics.MeasureString($variant, $legendFont)
        $graphics.DrawString($variant, $legendFont, [Drawing.Brushes]::Black, [single]($precisionX[$index]+34), [single](1390-$size.Height/2))
        $index++
    }
    $outputDirectory = Split-Path -Parent $Output
    if (-not (Test-Path -LiteralPath $outputDirectory)) { New-Item -ItemType Directory -Path $outputDirectory | Out-Null }
    $bitmap.Save($Output, [Drawing.Imaging.ImageFormat]::Png)
} finally {
    $centerFormat.Dispose(); $rightFormat.Dispose()
    $axisPen.Dispose(); $gridPen.Dispose(); $edgePen.Dispose(); $annotationPen.Dispose()
    $tickFont.Dispose(); $labelFont.Dispose(); $annotationFont.Dispose(); $legendFont.Dispose()
    $graphics.Dispose(); $bitmap.Dispose()
}
[pscustomobject]@{
    output = $Output; points = $points.Count
    deeplabv3 = @($points | Where-Object { $_.model -eq 'DeepLabV3 (ResNet34)' })
    non_dominated = $frontier
} | ConvertTo-Json -Depth 5
