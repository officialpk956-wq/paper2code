'use client';

import Link from 'next/link';
import { Upload, CheckCircle2, ArrowRight, Zap, Code2, Cpu } from 'lucide-react';

export default function ExtractCodePage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      {/* HERO SECTION */}
      <div className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#A78BFA]/20 via-[#0A0A0A] to-[#0A0A0A] opacity-50"></div>
        <div className="max-w-6xl mx-auto px-8 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#111] border border-[#262626] mb-8">
            <Zap size={14} className="text-[#A78BFA]" />
            <span className="text-[13px] font-medium text-[#A3A3A3]">New Feature</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
            Turn Research Papers<br/>Into <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#A78BFA] to-[#60A5FA]">Ready-To-Use Code</span>
          </h1>
          <p className="text-xl text-[#A3A3A3] mb-12 max-w-2xl mx-auto leading-relaxed">
            Upload any ML or AI paper in PDF format. We instantly parse the architecture, math, and concepts to generate complete, runnable PyTorch implementations.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/papers?tab=workspace" className="bg-[#A78BFA] text-black font-semibold text-[15px] px-8 py-4 rounded-xl hover:brightness-110 transition-all flex items-center gap-2">
              <Upload size={18} /> Upload Your First Paper
            </Link>
            <Link href="/papers" className="bg-[#111] border border-[#262626] text-white font-semibold text-[15px] px-8 py-4 rounded-xl hover:bg-[#1A1A1A] transition-all flex items-center gap-2">
              Browse the Library <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="py-24 bg-[#0D0D0D] border-y border-[#1A1A1A]">
        <div className="max-w-6xl mx-auto px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">How It Works</h2>
            <p className="text-[#A3A3A3]">From PDF to production-ready code in seconds.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connection Line */}
            <div className="hidden md:block absolute top-[60px] left-[15%] right-[15%] h-px bg-gradient-to-r from-transparent via-[#262626] to-transparent"></div>

            {/* Step 1 */}
            <div className="relative bg-[#111] border border-[#262626] rounded-2xl p-8 flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-[#1A1A1A] rounded-2xl flex items-center justify-center mb-6 border border-[#333] shadow-lg shadow-black/50 z-10">
                <Upload className="text-[#A78BFA]" size={28} />
              </div>
              <h3 className="text-lg font-bold mb-3">1. Upload PDF</h3>
              <p className="text-[14px] text-[#A3A3A3] leading-relaxed">
                Drop your arXiv PDF directly into our workspace. We support complex math rendering and multi-column layouts.
              </p>
            </div>

            {/* Step 2 */}
            <div className="relative bg-[#111] border border-[#262626] rounded-2xl p-8 flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-[#1A1A1A] rounded-2xl flex items-center justify-center mb-6 border border-[#333] shadow-lg shadow-black/50 z-10">
                <Cpu className="text-[#F97316]" size={28} />
              </div>
              <h3 className="text-lg font-bold mb-3">2. AI Extraction</h3>
              <p className="text-[14px] text-[#A3A3A3] leading-relaxed">
                Our vision models extract the exact neural architectures, hyperparameters, and pseudocode from the paper text.
              </p>
            </div>

            {/* Step 3 */}
            <div className="relative bg-[#111] border border-[#262626] rounded-2xl p-8 flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-[#1A1A1A] rounded-2xl flex items-center justify-center mb-6 border border-[#333] shadow-lg shadow-black/50 z-10">
                <Code2 className="text-[#4ADE80]" size={28} />
              </div>
              <h3 className="text-lg font-bold mb-3">3. Get Runnable Code</h3>
              <p className="text-[14px] text-[#A3A3A3] leading-relaxed">
                Receive a complete PyTorch implementation. You get a clean class structure, a forward pass, and training loop boilerplates.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* FEATURE DEMO UI */}
      <div className="py-24">
        <div className="max-w-6xl mx-auto px-8">
          <div className="bg-[#111] border border-[#262626] rounded-2xl overflow-hidden shadow-2xl flex flex-col lg:flex-row">
            <div className="w-full lg:w-1/2 p-12 bg-[#0D0D0D] flex flex-col justify-center border-b lg:border-b-0 lg:border-r border-[#262626]">
              <h2 className="text-2xl font-bold mb-4">Stop manually parsing complex equations.</h2>
              <p className="text-[#A3A3A3] text-[15px] leading-relaxed mb-8">
                Reading ML papers is hard. Re-implementing them is harder. Paper2Code handles the heavy lifting of extracting exact tensor shapes, normalization layers, and dropout probabilities, giving you a working template instantly.
              </p>
              <ul className="space-y-4">
                <li className="flex items-center gap-3 text-[14px] text-[#D4D4D4]">
                  <CheckCircle2 size={18} className="text-[#4ADE80]" /> Extracts explicit PyTorch network graphs
                </li>
                <li className="flex items-center gap-3 text-[14px] text-[#D4D4D4]">
                  <CheckCircle2 size={18} className="text-[#4ADE80]" /> Detects hyperparameter tables automatically
                </li>
                <li className="flex items-center gap-3 text-[14px] text-[#D4D4D4]">
                  <CheckCircle2 size={18} className="text-[#4ADE80]" /> Links code directly to equations in the paper
                </li>
              </ul>
              <Link href="/papers?tab=workspace" className="mt-8 text-[#A78BFA] font-medium flex items-center gap-2 hover:underline">
                Try it out in your Workspace <ArrowRight size={16} />
              </Link>
            </div>
            <div className="w-full lg:w-1/2 bg-[#1A1A1A] p-8 flex items-center justify-center">
              {/* Mock Code Window */}
              <div className="w-full max-w-md bg-[#0A0A0A] rounded-xl border border-[#333] shadow-2xl overflow-hidden">
                <div className="h-10 bg-[#111] border-b border-[#333] flex items-center px-4 gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#EF4444]"></div>
                  <div className="w-3 h-3 rounded-full bg-[#F59E0B]"></div>
                  <div className="w-3 h-3 rounded-full bg-[#10B981]"></div>
                  <div className="text-[11px] text-[#525252] font-mono ml-4">model.py</div>
                </div>
                <div className="p-4 font-mono text-[12px] leading-relaxed text-[#A3A3A3]">
                  <span className="text-[#F97316]">import</span> torch<br/>
                  <span className="text-[#F97316]">import</span> torch.nn <span className="text-[#F97316]">as</span> nn<br/><br/>
                  <span className="text-[#A78BFA]">class</span> <span className="text-[#60A5FA]">ExtractedModel</span>(nn.Module):<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-[#F97316]">def</span> <span className="text-[#60A5FA]">__init__</span>(self, hidden_dim=<span className="text-[#4ADE80]">768</span>):<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-[#525252]"># Extracted from Section 3.1</span><br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.attention = nn.MultiheadAttention(hidden_dim, <span className="text-[#4ADE80]">12</span>)<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.norm = nn.LayerNorm(hidden_dim)<br/><br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-[#F97316]">def</span> <span className="text-[#60A5FA]">forward</span>(self, x):<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out, _ = self.attention(x, x, x)<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-[#F97316]">return</span> self.norm(x + out)
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
