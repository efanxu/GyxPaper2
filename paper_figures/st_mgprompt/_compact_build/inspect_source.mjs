import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Final_Figures.pptx";
const outDir = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/_compact_build/source_inspect";

async function saveBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(outDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slides = presentation.slides.items;
const proto = presentation.toProto();
for (let i = 0; i < slides.length; i += 1) {
  await saveBlob(path.join(outDir, `slide-${i + 1}.png`), await presentation.export({ slide: slides[i], format: "png", scale: 2 }));
  await saveBlob(path.join(outDir, `slide-${i + 1}.layout.json`), await slides[i].export({ format: "layout" }));
}
const inspected = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,text,textPreview,textLines,bbox,bboxUnit,isPlaceholder,placeholders",
  maxChars: 200000,
});
await fs.writeFile(path.join(outDir, "source-inspect.ndjson"), inspected.ndjson ?? "", "utf8");
const help = presentation.help("*", {
  search: "group shape shapes.addGroup groupItems group collection slide.shapes.group",
  include: ["index", "examples", "notes"],
  maxChars: 30000,
});
await fs.writeFile(path.join(outDir, "api-help.ndjson"), help.ndjson ?? "", "utf8");
await fs.writeFile(path.join(outDir, "presentation-keys.txt"), [
  `presentation=${Object.keys(presentation).sort().join(",")}`,
  `slides=${Object.keys(presentation.slides).sort().join(",")}`,
  `slide0=${Object.keys(slides[0]).sort().join(",")}`,
  `shapes0=${Object.keys(slides[0].shapes).sort().join(",")}`,
  `shape0=${Object.keys(slides[0].shapes.items[0] ?? {}).sort().join(",")}`,
  `slideSize=${JSON.stringify(presentation.slideSize)}`,
  `masters=${presentation.masters?.items?.length ?? "?"}`,
  `layouts=${presentation.layouts?.items?.length ?? "?"}`,
  `presentationProto=${Object.getOwnPropertyNames(Object.getPrototypeOf(presentation)).sort().join(",")}`,
  `slidesProto=${Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)).sort().join(",")}`,
  `slideProto=${Object.getOwnPropertyNames(Object.getPrototypeOf(slides[0])).sort().join(",")}`,
  `shapesProto=${Object.getOwnPropertyNames(Object.getPrototypeOf(slides[0].shapes)).sort().join(",")}`,
  `shapeProto=${Object.getOwnPropertyNames(Object.getPrototypeOf(slides[0].shapes.items[0])).sort().join(",")}`,
  `protoKeys=${Object.keys(proto).sort().join(",")}`,
  `protoSummary=${JSON.stringify(Object.fromEntries(Object.entries(proto).filter(([k]) => !["slides", "slideMasters", "slideLayouts", "theme"].includes(k))))}`,
  `slideProto0Keys=${Object.keys(proto.slides?.[0] ?? {}).sort().join(",")}`,
  `slideProto0Frame=${JSON.stringify(proto.slides?.[0]?.frame ?? proto.slides?.[0]?.size ?? proto.slides?.[0]?.slideSize ?? null)}`,
].join("\n"), "utf8");
console.log(`slides=${slides.length}`);
console.log(`slideSize=${JSON.stringify(presentation.slideSize)}`);
console.log(`out=${outDir}`);
