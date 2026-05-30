import React from 'react';
import Editor from '@monaco-editor/react';

interface CodeEditorProps {
  code: string;
  onChange: (value: string | undefined) => void;
  language?: string;
}

export default function CodeEditor({ code, onChange, language = "python" }: CodeEditorProps) {
  return (
    <div className="w-full h-full rounded-lg overflow-hidden border border-gray-700 shadow-xl">
      <Editor
        height="100%"
        defaultLanguage={language}
        theme="vs-dark"
        value={code}
        onChange={onChange}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: "on",
          roundedSelection: false,
          scrollBeyondLastLine: false,
          readOnly: false,
          automaticLayout: true,
        }}
      />
    </div>
  );
}
