# app/services/report/pdf_generator.py
import os
import datetime
import markdown2
import pdfkit
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from ...utils.config import Config
from ...utils.constants import NOTICE
from ...models.schemas import AgentState

class PDFGenerator:
    """PDF 보고서 생성 서비스"""
    
    def __init__(self):
        Config.create_directories()
    
    def generate_pdf(self, state: AgentState) -> AgentState:
        """PDF 보고서 생성"""
        startup_name = state.get("startup_name", "알 수 없는 스타트업")
        today = datetime.datetime.now().strftime("%Y%m%d")
        
        # 파일 경로 설정
        md_path = Config.OUTPUT_DIR / f"{startup_name}_보고서.md"
        pdf_path = Config.OUTPUT_DIR / f"{startup_name}_투자분석보고서_{today}.pdf"
        
        try:
            # Markdown 생성
            self._generate_markdown(state, md_path)
            
            # PDF 변환
            self._convert_md_to_pdf(md_path, pdf_path)
            
            state["pdf_path"] = str(pdf_path)
            print(f"PDF 생성 완료: {pdf_path}")
            
        except Exception as e:
            print(f"PDF 생성 오류: {e}")
            # ReportLab으로 백업 생성
            self._generate_reportlab_pdf(state, pdf_path)
            state["pdf_path"] = str(pdf_path)
        
        return state
    
    def _generate_markdown(self, state: dict, md_path) -> None:
        """Markdown 보고서 생성"""
        sn = state.get("startup_name", "알 수 없는 스타트업")
        rep = state.get("보고서", "보고서 내용이 없습니다.")
        dec = state.get("최종_판단", "보류")

        labels = ["상품/서비스", "기술", "성장률", "시장성", "경쟁사"]
        keys = ["상품_점수", "기술_점수", "성장률_점수", "시장성_점수", "경쟁사_점수"]
        scores = [int(state.get(k, 0)) for k in keys]
        avg = sum(scores) / len(scores)

        dec_html = ("<span style='color:green;'>투자</span>"
                    if dec == "투자" else "<span style='color:red;'>보류</span>")

        md = [
            f"<h1 align='center'>{sn} 투자 분석 보고서</h1>", "",
            f"- **작성일** : {datetime.datetime.now():%Y년 %m월 %d일}",
            f"- **최종 판단** : {dec_html}", "", "---", "",
            "## 1. 점수 요약", "",
            "| 평가 항목 | 점수 |", "|:-----------:|:----:|",
            *[f"| {l} | **{v}** |" for l, v in zip(labels, scores)],
            f"| **평균** | **{avg:.1f}** |", "",
            f"> **{NOTICE}**",
            "", "---", "",
            "## 2. 상세 분석", ""
        ]

        # 상세 분석 추가
        for line in rep.strip().splitlines():
            if line.startswith("### "):
                md.append(f"#### **{line[4:]}**")
            elif line.startswith("## "):
                md.append(f"### **{line[3:]}**")
            elif line.startswith("# "):
                md.append(f"## **{line[2:]}**")
            else:
                md.append(line)

        md += ["", "---", "*이 보고서는 AI 분석 시스템에 의해 자동 생성되었습니다.*"]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    def _convert_md_to_pdf(self, md_path, pdf_path) -> None:
        """Markdown을 PDF로 변환"""
        wkhtmltopdf_path = Config.BASE_DIR / "wkhtmltopdf" / "bin" / "wkhtmltopdf.exe"
        
        if not wkhtmltopdf_path.exists():
            raise FileNotFoundError("wkhtmltopdf가 설치되지 않았습니다.")
        
        cfg = pdfkit.configuration(wkhtmltopdf=str(wkhtmltopdf_path))
        
        with open(md_path, "r", encoding="utf-8") as f:
            body_html = markdown2.markdown(f.read(), extras=["tables"])

        # 폰트 CSS 설정
        if Config.NANUM_REG_TTF.exists():
            reg = str(Config.NANUM_REG_TTF).replace("\\", "/")
            bold = str(Config.NANUM_BOLD_TTF if Config.NANUM_BOLD_TTF.exists() else Config.NANUM_REG_TTF).replace("\\", "/")
            font_css = (f"@font-face{{font-family:'NanumGothic';src:url('file:///{reg}') format('truetype');font-weight:normal;}}"
                        f"@font-face{{font-family:'NanumGothic';src:url('file:///{bold}') format('truetype');font-weight:bold;}}")
            family = "NanumGothic"
        else:
            font_css, family = "", "sans-serif"

        style = f"""
        <style>
        {font_css}
        body{{margin:40px 50px 60px;font-family:'{family}';line-height:1.6;font-size:11pt}}
        h1{{font-size:20pt;text-align:center;margin-bottom:0.6em}}
        h2{{font-size:15pt;margin-top:1.5em;margin-bottom:0.4em}}
        table{{border-collapse:collapse;width:100%;margin-top:0.8em;font-size:10.5pt}}
        th,td{{border:1px solid #666;padding:6px 8px;text-align:center}}
        th{{background:#e0e0e0;font-weight:bold}}
        tr:last-child td{{background:#f5f5f5;font-weight:bold}}
        </style>"""

        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{style}</head><body>{body_html}</body></html>"
        pdfkit.from_string(html, str(pdf_path), configuration=cfg,
                           options={"enable-local-file-access": None, "encoding": "utf-8"})

    def _generate_reportlab_pdf(self, state: dict, pdf_path) -> None:
        """ReportLab으로 백업 PDF 생성"""
        sn = state.get("startup_name", "알 수 없는 스타트업")
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # 폰트 설정
        try:
            pdfmetrics.registerFont(TTFont("NanumGothic", str(Config.NANUM_REG_TTF)))
            pdfmetrics.registerFont(TTFont("NanumGothic-Bold", str(Config.NANUM_BOLD_TTF)))
            base, bold = "NanumGothic", "NanumGothic-Bold"
        except Exception:
            base, bold = "Helvetica", "Helvetica-Bold"

        # 스타일 추가
        styles.add(ParagraphStyle("ReportTitle", parent=styles["Heading1"], 
                                fontSize=18, alignment=1, spaceAfter=20, fontName=base))
        styles.add(ParagraphStyle("ReportSubtitle", parent=styles["Heading2"], 
                                fontSize=14, spaceBefore=10, spaceAfter=10, fontName=base))

        elems = [
            Paragraph(f"{sn} 투자 분석 보고서", styles["ReportTitle"]),
            Paragraph(f"작성일: {datetime.datetime.now():%Y년 %m월 %d일}", styles["Normal"]),
            Spacer(1, 20)
        ]

        # 점수 테이블
        labels = ["상품/서비스", "기술", "성장률", "시장성", "경쟁사"]
        keys = ["상품_점수", "기술_점수", "성장률_점수", "시장성_점수", "경쟁사_점수"]
        scores = [int(state.get(k, 0)) for k in keys]
        avg = sum(scores) / len(scores)

        t_data = [["평가 항목", "점수"]] + [[l, str(s)] for l, s in zip(labels, scores)] + [["평균", f"{avg:.1f}"]]
        t = Table(t_data, colWidths=[300, 100])
        
        elems.extend([
            Paragraph("점수 요약", styles["ReportSubtitle"]),
            t,
            Spacer(1, 20),
            Paragraph("상세 보고서는 웹 인터페이스에서 확인하세요.", styles["Normal"])
        ])

        doc.build(elems)