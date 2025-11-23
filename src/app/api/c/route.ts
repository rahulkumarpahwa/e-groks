import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const GET = async () => {
  try {
    // Resolve the path to the `ai.py` file
    const filePath = path.join(process.cwd(), "src", "util", "programs.c");

    // Read the file content
    const fileContent = await fs.readFile(filePath, "utf-8");

    // Return the file content as plain text
    return new NextResponse(fileContent, {
      headers: {
        "Content-Type": "text/plain",
      },
    });
  } catch (error: unknown) {
    console.error("Error reading file:", (error as Error).message || error);
    return NextResponse.json(
      { message: "Failed to fetch file", success: false },
      { status: 500 }
    );
  }
};