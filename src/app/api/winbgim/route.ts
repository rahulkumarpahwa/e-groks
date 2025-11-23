import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "src", "util", "winbgim.h");
    
    const fileContent = await fs.readFile(filePath, "utf-8");
    
    return new NextResponse(fileContent, {
      status: 200,
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": 'inline; filename="graphics.h"',
      },
    });
  } catch (error) {
    console.error("Error reading graphics.h:", error);
    
    return NextResponse.json(
      {
        error: "Failed to read graphics.h file",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}