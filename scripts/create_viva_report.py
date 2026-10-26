"""Build the printable progress report from its reviewed Markdown source."""
from pathlib import Path
import base64, html, json, re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
source_path=ROOT/'docs/PROJECT-PROGRESS-AND-VIVA.md'
source=source_path.read_text(encoding='utf8')
market=json.loads((ROOT/'public/data/market.json').read_text(encoding='utf8'))
names={'Last 28 days':'Last 28 days', '84-day daily mean':'84-day mean', 'Damped trend':'Damped trend',
       'Gradient boosting (log units)':'Existing log boosting', 'Gradient boosting (Poisson)':'Base Poisson boosting',
       'History boosting (absolute error)':'History absolute error', 'History boosting (log units)':'History log boosting',
       'History boosting (Poisson)':'History Poisson boosting', 'History extra trees':'History extra trees'}
rows=['| Method | Validation MAE | Test MAE | Test WAPE percent | Test RMSE |',
      '| --- | --- | --- | --- | --- |']
for row in market['demand']['candidates']:
    rows.append(f"| {names[row['model']]} | {row['validation']['mae']:.2f} | {row['test']['mae']:.2f} | {100*row['test']['wape']:.2f} | {row['test']['rmse']:.2f} |")
source=re.sub(r'<!--MODEL_TABLE_START-->.*?<!--MODEL_TABLE_END-->',
              '<!--MODEL_TABLE_START-->\n'+'\n'.join(rows)+'\n<!--MODEL_TABLE_END-->',source,flags=re.S)
source_path.write_text(source,encoding='utf8')
output=ROOT/'output/pdf/SmartRetail_Project_Progress_and_Viva.pdf'
output.parent.mkdir(parents=True,exist_ok=True)
qa=ROOT/'artifacts/report-qa'; qa.mkdir(parents=True,exist_ok=True)

font_dir=Path('C:/Windows/Fonts')
for name,file in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf'),('ArialItalic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='ArialItalic',boldItalic='ArialBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='ArialBold',fontSize=22,leading=27,spaceAfter=12,textColor=colors.black))
styles.add(ParagraphStyle(name='ReportH1',fontName='ArialBold',fontSize=14,leading=18,spaceBefore=17,spaceAfter=7,keepWithNext=True,textColor=colors.black))
styles.add(ParagraphStyle(name='ReportH2',fontName='ArialBold',fontSize=11,leading=15,spaceBefore=10,spaceAfter=5,keepWithNext=True,textColor=colors.black))
styles.add(ParagraphStyle(name='ReportBody',fontName='Arial',fontSize=10.2,leading=14.4,spaceAfter=7,textColor=colors.black))
styles.add(ParagraphStyle(name='ReportBullet',parent=styles['ReportBody'],leftIndent=11,firstLineIndent=0,bulletIndent=0,spaceAfter=5))
styles.add(ParagraphStyle(name='ReportCell',fontName='Arial',fontSize=9,leading=12,spaceAfter=0))
styles.add(ParagraphStyle(name='ReportCellHead',parent=styles['ReportCell'],fontName='ArialBold',textColor=colors.white))
styles.add(ParagraphStyle(name='ReportCaption',fontName='Arial',fontSize=8.5,leading=11,textColor=colors.HexColor('#444444'),spaceAfter=7))

def normalize(text):
    return text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('→','->').replace('≥','>=').replace('≤','<=')

def inline(text):
    text=html.escape(normalize(text))
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#254d3d">\1</link>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    return re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)

width=A4[0]-100
story=[]; lines=source.splitlines(); index=0
while index<len(lines):
    line=lines[index].strip()
    if not line or line.startswith('<!--'):
        index+=1; continue
    if line.startswith('# '):
        story.append(Paragraph(inline(line[2:]),styles['ReportTitle']))
    elif line.startswith('## '):
        if line.startswith('## 13 '): story.append(PageBreak())
        story.append(Paragraph(inline(line[3:]),styles['ReportH1']))
    elif line.startswith('### '):
        story.append(Paragraph(inline(line[4:]),styles['ReportH2']))
    elif line.startswith('|'):
        table_lines=[]
        while index<len(lines) and lines[index].strip().startswith('|'):
            candidate=lines[index].strip()
            if not re.fullmatch(r'[|\s:-]+',candidate):
                table_lines.append([part.strip() for part in candidate.strip('|').split('|')])
            index+=1
        ncols=len(table_lines[0])
        fractions={2:[.70,.30],3:[.24,.54,.22],5:[.38,.16,.15,.16,.15]}[ncols]
        data=[[Paragraph(inline(text),styles['ReportCellHead'] if row==0 else styles['ReportCell']) for text in cells]
              for row,cells in enumerate(table_lines)]
        table=Table(data,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#284d40')),
                                   ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9d9d9')),
                                   ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                   ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                                   ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                                   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f5f3')])]))
        story.extend([Spacer(1,3),KeepTogether([table]) if ncols==2 else table,Spacer(1,10)])
        if ncols==5:
            notebook=json.loads((ROOT/'notebooks/SmartRetail_Presentation_Colab.ipynb').read_text(encoding='utf8'))
            image_data=next(o['data']['image/png'] for c in notebook['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
            chart=qa/'model-results.png'; chart.write_bytes(base64.b64decode(image_data))
            picture=Image(str(chart)); factor=width/picture.imageWidth
            picture.drawWidth=width; picture.drawHeight=picture.imageHeight*factor
            story.extend([picture,Paragraph('Figure 1  Three-method presentation comparison on the reused 12 November to 9 December 2011 benchmark. Source: executed presentation notebook.',styles['ReportCaption'])])
        continue
    elif line.startswith('- '):
        story.append(Paragraph(inline(line[2:]),styles['ReportBullet'],bulletText='-'))
    else:
        story.append(Paragraph(inline(line),styles['ReportBody']))
    index+=1

def footer(canvas,document):
    canvas.setFont('Arial',8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(50,28,'SmartRetail  |  Academic progress and viva  |  26 October 2026')
    canvas.drawRightString(A4[0]-50,28,str(document.page))

document=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=48,
                           title='SmartRetail Project Progress Report and Viva Guide',author='SmartRetail project')
document.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created',output,'| bytes',output.stat().st_size)
