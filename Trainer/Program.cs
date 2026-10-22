using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.ML;
using Microsoft.ML.Data;
using Normalizer;

// Trainer <data.json | folder> <version> <output directory>
// Writes spam.mlnet, manifest.json, and gheychi-model-v{version}.zip the Gheychi app reads to load it.
if (args.Length != 3 || !int.TryParse(args[1], NumberStyles.None, CultureInfo.InvariantCulture, out var version) || version <= 0)
{
    Console.Error.WriteLine("Usage: Trainer <data.json | folder> <version> <output directory>");
    return 2;
}

const string ModelFile = "spam.mlnet";
const string SpamLabel = "spam";
const string HamLabel = "ham";
const int Folds = 5;
const int Seed = 1;

// The app's default spam threshold (SpamDetector.DefaultThreshold). The gate is about what users see at that setting:
// a real message hidden as spam costs far more than an ad that reaches the inbox.
const float AppThreshold = 0.85f;
const double MinMacroAccuracy = 0.90;
const double MaxRealHiddenRate = 0.04;

var inputPath = args[0];
var outputDirectory = args[2];

// A folder input allows combining official data with user exports (gheychi-*.json). Later files take priority;
// non-gheychi files are read first, then gheychi exports in alphabetical (chronological) order.
List<(string Path, int Priority)> files;
if (Directory.Exists(inputPath))
{
    var nonGheychi = Directory.GetFiles(inputPath, "*.json", SearchOption.TopDirectoryOnly)
        .Where(f => !Path.GetFileName(f).StartsWith("gheychi-", StringComparison.Ordinal))
        .OrderBy(Path.GetFileName, StringComparer.Ordinal);

    var gheychi = Directory.GetFiles(inputPath, "*.json", SearchOption.TopDirectoryOnly)
        .Where(f => Path.GetFileName(f).StartsWith("gheychi-", StringComparison.Ordinal))
        .OrderBy(Path.GetFileName, StringComparer.Ordinal);

    files = nonGheychi.Concat(gheychi)
        .Select((path, index) => (Path: path, Priority: index))
        .ToList();
}
else
{
    files = [(inputPath, 0)];
}

var totalItems = 0;
var rawRows = new List<PrioritizedRow>();

foreach (var (filePath, priority) in files)
{
    var items = JsonSerializer.Deserialize<List<DatasetItem>>(File.ReadAllText(filePath)) ?? [];
    Console.WriteLine($"Read {items.Count} rows from {filePath}");
    totalItems += items.Count;
    foreach (var item in items)
    {
        var normalized = TextNormalizer.Normalize(item.Text);
        if (normalized.Length > 0)
        {
            rawRows.Add(new PrioritizedRow(normalized, item.Label == 1 ? SpamLabel : HamLabel, priority));
        }
    }
}

// The app scores normalized text, so the model is trained on exactly what TextNormalizer produces.
// If files disagree on a text's label, the highest-priority file wins; conflict inside that file drops the text.
var overridden = 0;
var rows = new List<Row>();

foreach (var group in rawRows.GroupBy(r => r.Text))
{
    var highestPriority = group.Max(r => r.Priority);
    var top = group.Where(r => r.Priority == highestPriority).ToList();
    var topLabels = top.Select(r => r.Label).Distinct().ToList();

    if (topLabels.Count > 1)
    {
        continue;
    }

    rows.Add(new Row { Text = group.Key, Label = topLabels[0] });

    if (group.Select(r => r.Label).Distinct().Count() > 1)
    {
        overridden++;
    }
}

Console.WriteLine($"{overridden} texts took the label of a newer file");

var spamCount = rows.Count(row => row.Label == SpamLabel);
Console.WriteLine($"{totalItems} items, {rows.Count} distinct usable rows ({spamCount} spam, {rows.Count - spamCount} ham)");

var ml = new MLContext(Seed);
var data = ml.Data.LoadFromEnumerable(rows);

// Cross-validation rather than one holdout: on a few thousand rows a single 10% split moves by more than a point
// from one seed to the next, which hides the changes worth measuring.
var folds = ml.MulticlassClassification.CrossValidate(data, BuildPipeline(ml), Folds, seed: Seed);
var macro = folds.Select(f => f.Metrics.MacroAccuracy).ToList();
var macroMean = macro.Average();
var macroSd = Math.Sqrt(macro.Sum(m => (m - macroMean) * (m - macroMean)) / (macro.Count - 1));

int realHidden = 0, realTotal = 0, spamCaught = 0, spamTotal = 0;
foreach (var fold in folds)
{
    foreach (var (score, isSpam) in SpamScores(ml, fold.ScoredHoldOutSet))
    {
        if (isSpam)
        {
            spamTotal++;
            if (score >= AppThreshold) spamCaught++;
        }
        else
        {
            realTotal++;
            if (score >= AppThreshold) realHidden++;
        }
    }
}
var realHiddenRate = realHidden / (double)realTotal;
var spamCaughtRate = spamCaught / (double)spamTotal;

Console.WriteLine($"{Folds}-fold macro accuracy {macroMean:F4} (SD {macroSd:F4}; folds {string.Join(" ", macro.Select(m => m.ToString("F4", CultureInfo.InvariantCulture)))})");
Console.WriteLine($"At threshold {AppThreshold}: real messages hidden {realHidden} of {realTotal} ({realHiddenRate:P2}), spam caught {spamCaught} of {spamTotal} ({spamCaughtRate:P2})");

if (macroMean < MinMacroAccuracy || realHiddenRate > MaxRealHiddenRate)
{
    Console.Error.WriteLine($"Not publishing: needs macro accuracy >= {MinMacroAccuracy:P0} and at most {MaxRealHiddenRate:P0} of real messages hidden at {AppThreshold}.");
    return 1;
}

// The released model learns from every row; the folds above only measured how well this pipeline generalizes.
var model = BuildPipeline(ml).Fit(data);

Directory.CreateDirectory(outputDirectory);
var modelPath = Path.Combine(outputDirectory, ModelFile);
ml.Model.Save(model, data.Schema, modelPath);
VerifyLoads(modelPath);

var manifest = new JsonObject
{
    ["version"] = version,
    ["model"] = ModelFile,
    ["textColumn"] = nameof(Row.Text),
    ["labelColumn"] = nameof(Row.Label),
    ["spamLabel"] = SpamLabel,
    ["normalizer"] = TextNormalizer.Version,
    ["size"] = new FileInfo(modelPath).Length,
    ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath))),
    ["metrics"] = new JsonObject
    {
        ["rows"] = rows.Count,
        ["spamRows"] = spamCount,
        ["folds"] = Folds,
        ["macroAccuracy"] = Math.Round(macroMean, 4),
        ["macroAccuracySd"] = Math.Round(macroSd, 4),
        ["threshold"] = AppThreshold,
        ["realHiddenRate"] = Math.Round(realHiddenRate, 4),
        ["spamCaughtRate"] = Math.Round(spamCaughtRate, 4)
    }
};
var manifestPath = Path.Combine(outputDirectory, "manifest.json");
File.WriteAllText(manifestPath, manifest.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

var zipPath = Path.Combine(outputDirectory, $"gheychi-model-v{version}.zip");
if (File.Exists(zipPath))
{
    File.Delete(zipPath);
}
using (var archive = ZipFile.Open(zipPath, ZipArchiveMode.Create))
{
    archive.CreateEntryFromFile(manifestPath, "manifest.json");
    archive.CreateEntryFromFile(modelPath, ModelFile);
}

Console.WriteLine($"Model v{version} written to {outputDirectory}");
return 0;

// A linear model over the text features. Tree ensembles scored most messages at exactly 0 or 1, so the app's
// threshold setting barely changed anything; this one gives graded scores and hid half as many real messages at
// the same threshold in cross-validation. L2 0.03 was the best of the values tried.
static IEstimator<ITransformer> BuildPipeline(MLContext ml) =>
    ml.Transforms.Text.FeaturizeText("Features", nameof(Row.Text))
        .Append(ml.Transforms.Conversion.MapValueToKey(nameof(Row.Label)))
        .Append(ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
            labelColumnName: nameof(Row.Label),
            featureColumnName: "Features",
            l1Regularization: 0,
            l2Regularization: 0.03f))
        .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// The scored label column is a key: its value is the 1-based index of the label among the key values.
static IEnumerable<(float Score, bool IsSpam)> SpamScores(MLContext ml, IDataView scored)
{
    var labels = default(VBuffer<ReadOnlyMemory<char>>);
    scored.Schema[nameof(Row.Label)].GetKeyValues(ref labels);
    var spamIndex = labels.DenseValues().Select(v => v.ToString()).ToList().IndexOf(SpamLabel);
    foreach (var p in ml.Data.CreateEnumerable<ScoredRow>(scored, false))
        yield return (p.Score[spamIndex], p.Label == spamIndex + 1);
}

// The app finds the spam score by the label's position among the model's key values; check it can.
static void VerifyLoads(string modelPath)
{
    var ml = new MLContext();
    var model = ml.Model.Load(modelPath, out _);
    var engine = ml.Model.CreatePredictionEngine<Row, Prediction>(model);
    var labels = default(VBuffer<ReadOnlyMemory<char>>);
    engine.OutputSchema[nameof(Row.Label)].GetKeyValues(ref labels);
    var names = labels.DenseValues().Select(v => v.ToString()).ToList();
    if (!names.Contains(SpamLabel) || engine.Predict(new Row()).Score.Length != names.Count)
        throw new InvalidDataException($"The saved model's labels ({string.Join(", ", names)}) do not include '{SpamLabel}'.");
}

sealed record DatasetItem(string Text, int Label);

sealed record PrioritizedRow(string Text, string Label, int Priority);

sealed class Row
{
    public string Text { get; set; } = "";
    public string Label { get; set; } = "";
}

sealed class Prediction
{
    public float[] Score { get; set; } = [];
}

sealed class ScoredRow
{
    public uint Label { get; set; }
    public float[] Score { get; set; } = [];
}
