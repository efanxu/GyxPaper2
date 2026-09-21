import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const p = await PresentationFile.importPptx(await FileBlob.load("D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Final_Figures.pptx"));
for (const q of ["setViewportSize", "slides.remove slide.delete", "fontSizePt typeface text style", "group shapes"]) {
  const h = p.help("*", { search: q, include: ["index", "examples", "notes"], maxChars: 12000 });
  console.log(`QUERY ${q}\n${h.ndjson ?? ""}`);
}
