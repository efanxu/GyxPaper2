import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Fig2_Fig3_Compact_Final.pptx";
const outDir = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/_figure2_redesign_build/source_inspect";
async function save(file, blob) { await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer())); }
await fs.mkdir(outDir, { recursive: true });
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const slides = p.slides.items;
for (let i = 0; i < slides.length; i += 1) {
  await save(path.join(outDir, `slide-${i + 1}.png`), await p.export({ slide: slides[i], format: "png", scale: 2 }));
  await save(path.join(outDir, `slide-${i + 1}.layout.json`), await slides[i].export({ format: "layout" }));
}
const inspect = await p.inspect({ kind: "deck,slide,textbox,shape,image,table,chart,notes,layout", include: "id,slide,name,title,text,textPreview,textChars,textLines,bbox,bboxUnit,isPlaceholder,placeholders", maxChars: 200000 });
await fs.writeFile(path.join(outDir, "source-inspect.ndjson"), inspect.ndjson ?? "", "utf8");
const records = (inspect.ndjson ?? "").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
const summary = slides.map((slide, index) => {
  const textRecords = records.filter((r) => r.slide === index + 1 && r.kind === "textbox");
  return {
    slide: index + 1,
    widthPx: slide.toProto().widthEmu / 9525,
    heightPx: slide.toProto().heightEmu / 9525,
    textBoxes: textRecords.length,
    textChars: textRecords.reduce((sum, r) => sum + (r.textChars ?? (r.text ?? "").length), 0),
    placeholders: records.filter((r) => r.slide === index + 1 && r.isPlaceholder === true).length,
  };
});
await fs.writeFile(path.join(outDir, "source-summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(JSON.stringify(summary));
