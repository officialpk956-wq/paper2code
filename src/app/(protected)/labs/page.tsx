'use client';

import Link from 'next/link';
import { FadeIn } from '@/components/FadeIn';
import AnimatedCard from '@/components/AnimatedCard';

type Lab = { id: string; archSlug: string; title: string; subtitle: string; color: string; badge: string; svgContent: React.ReactNode; };

const TRANSFORMER_SVG = (
  <svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    {[[20,35,50,60],[80,35,50,60],[140,35,50,60]].map(([x,y,w,h],i) => (
      <g key={i}>
        <rect x={x} y={y} width={w} height={h} rx={6} fill={'#A78BFA'+(i===1?'30':'18')} stroke={'#A78BFA'+(i===1?'80':'40')} strokeWidth={1}/>
        <text x={x+w/2} y={y+h/2+1} textAnchor="middle" dominantBaseline="middle" fill="#A78BFA" fontSize={8} fontWeight={600}>
          {['Q','K','V'][i]}
        </text>
      </g>
    ))}
    <text x={100} y={20} textAnchor="middle" fill="#A78BFA" fontSize={9} fontWeight={700}>Multi-Head Attention</text>
    <line x1={45} y1={95} x2={100} y2={115} stroke="#A78BFA" strokeWidth={1} opacity={0.5}/>
    <line x1={105} y1={95} x2={100} y2={115} stroke="#A78BFA" strokeWidth={1} opacity={0.5}/>
    <line x1={165} y1={95} x2={100} y2={115} stroke="#A78BFA" strokeWidth={1} opacity={0.5}/>
    <rect x={60} y={115} width={80} height={14} rx={4} fill="#A78BFA20" stroke="#A78BFA50" strokeWidth={1}/>
    <text x={100} y={122} textAnchor="middle" dominantBaseline="middle" fill="#A78BFA" fontSize={8}>Concat + Linear</text>
  </svg>
);

const CNN_SVG = (
  <svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <text x={100} y={15} textAnchor="middle" fill="#60A5FA" fontSize={9} fontWeight={700}>CNN Forward Pass</text>
    {[[10,25,24,80],[45,30,24,70],[80,25,28,80],[120,35,24,60],[155,28,24,74]].map(([x,y,w,h],i) => (
      <g key={i}>
        <rect x={x} y={y} width={w} height={h} rx={3} fill={'#60A5FA'+(i<3?'20':'14')} stroke={'#60A5FA'+(i<3?'70':'40')} strokeWidth={1}/>
        {i < 3 && <text x={x+w/2} y={y-4} textAnchor="middle" fill="#60A5FA80" fontSize={7}>C{i+1}</text>}
      </g>
    ))}
    <text x={100} y={120} textAnchor="middle" fill="#60A5FA80" fontSize={8}>Feature Maps</text>
  </svg>
);

const VIT_SVG = (
  <svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <text x={100} y={12} textAnchor="middle" fill="#A78BFA" fontSize={9} fontWeight={700}>Vision Transformer</text>
    {[0,1,2,3,4,5,6,7,8].map(i => {
      const col = i % 3, row = Math.floor(i / 3);
      return <rect key={i} x={15+col*40} y={18+row*30} width={36} height={26} rx={4}
        fill={'#A78BFA'+(i===4?'35':'18')} stroke={'#A78BFA'+(i===4?'80':'40')} strokeWidth={1}/>;
    })}
    <text x={15+1*40+18} y={18+1*30+13} textAnchor="middle" dominantBaseline="middle" fill="#A78BFA" fontSize={7}>[CLS]</text>
    {[0,1,2,5,6,7,8].map(i => {
      const col = i%3, row = Math.floor(i/3);
      return <text key={i} x={15+col*40+18} y={18+row*30+13} textAnchor="middle" dominantBaseline="middle" fill="#A78BFA70" fontSize={7}>P{i}</text>;
    })}
    <line x1={100} y1={108} x2={100} y2={118} stroke="#A78BFA" strokeWidth={1}/>
    <rect x={60} y={118} width={80} height={12} rx={4} fill="#A78BFA20" stroke="#A78BFA50" strokeWidth={1}/>
    <text x={100} y={124} textAnchor="middle" dominantBaseline="middle" fill="#A78BFA" fontSize={7}>Transformer Encoder</text>
  </svg>
);

const DIFFUSION_SVG = (
  <svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <text x={100} y={12} textAnchor="middle" fill="#F472B6" fontSize={9} fontWeight={700}>Diffusion Process</text>
    {[0,1,2,3,4].map(i => {
      const x = 10 + i * 38;
      const opacity = 0.2 + i * 0.2;
      return (
        <g key={i}>
          <rect x={x} y={22} width={32} height={80} rx={6}
            fill={`rgba(244,114,182,${opacity * 0.3})`}
            stroke={`rgba(244,114,182,${opacity})`} strokeWidth={1}/>
          {i < 4 && <line x1={x+32} y1={62} x2={x+38} y2={62} stroke="#F472B680" strokeWidth={1}
            markerEnd="url(#da)"/>}
          <text x={x+16} y={112} textAnchor="middle" fill="#F472B680" fontSize={7}>t={1-i*0.25}</text>
        </g>
      );
    })}
    <defs>
      <marker id="da" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto">
        <path d="M0,0 L0,4 L4,2z" fill="#F472B680"/>
      </marker>
    </defs>
  </svg>
);

const LABS: Lab[] = [
  { id: 'transformer', archSlug: 'transformer',     title: 'Transformer',       subtitle: 'Explore multi-head attention, positional encodings, and the full encoder-decoder stack.',  color: '#A78BFA', badge: 'NLP',    svgContent: TRANSFORMER_SVG },
  { id: 'cnn',         archSlug: 'resnet',           title: 'CNN',               subtitle: 'Visualize feature maps, convolution kernels, and how pooling layers downsample.',           color: '#60A5FA', badge: 'Vision', svgContent: CNN_SVG },
  { id: 'vit',         archSlug: 'vit',              title: 'Vision Transformer', subtitle: 'Watch patches become tokens and see how a CNN-free model processes images.',               color: '#A78BFA', badge: 'Vision', svgContent: VIT_SVG },
  { id: 'diffusion',   archSlug: 'stable-diffusion', title: 'Diffusion',         subtitle: 'Step through forward noise addition and reverse denoising in a latent diffusion model.',   color: '#F472B6', badge: 'Gen.',   svgContent: DIFFUSION_SVG },
];

export default function LabsPage() {
  return (
    <div className="min-h-screen bg-transparent text-white">
      {/* HEADER */}
      <div className="border-b border-[#1A1A1A] bg-[#0A0A0A] px-8 py-6">
        <h1 className="text-[26px] font-bold text-white">AI Labs</h1>
        <p className="mt-1 text-[13px] text-[#525252]">
          Visual guides to the architectures behind modern AI — interactive parameter labs are on the roadmap.
        </p>
      </div>

      {/* GRID */}
      <div className="mx-auto max-w-6xl px-8 py-8 grid grid-cols-1 md:grid-cols-2 gap-5">
        {LABS.map((lab, idx) => (
          <FadeIn key={lab.id} delay={idx * 0.1}>
            <AnimatedCard>
              <Link href={`/architectures/${lab.archSlug}`}
                className="group rounded-2xl border border-[#262626] bg-[#111111] overflow-hidden transition-colors hover:border-[#A78BFA]/30 block h-full">
                {/* Preview */}
                <div className="h-44 bg-[#0A0A0A] flex items-center justify-center p-4">
                  {lab.svgContent}
                </div>
                <div className="p-5">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="h-0.5 w-6 rounded-full" style={{ backgroundColor: lab.color }} />
                    <span className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                      style={{ backgroundColor: lab.color + '20', color: lab.color }}>
                      {lab.badge}
                    </span>
                  </div>
                  <div className="text-[16px] font-bold text-white">{lab.title}</div>
                  <div className="mt-1 text-[12px] leading-relaxed text-[#A3A3A3]">{lab.subtitle}</div>
                  <div className="mt-4 text-[12px] font-semibold group-hover:underline" style={{ color: lab.color }}>
                    Explore the Architecture →
                  </div>
                </div>
              </Link>
            </AnimatedCard>
          </FadeIn>
        ))}
      </div>
    </div>
  );
}
