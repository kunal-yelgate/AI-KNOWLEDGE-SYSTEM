import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const sessions = await prisma.chatSession.findMany({
      where: { userId: session.user.id },
      orderBy: { updatedAt: "desc" },
      include: {
        document: {
          select: { id: true, filename: true },
        },
        _count: {
          select: { messages: true },
        },
      },
    });

    return NextResponse.json({ sessions });
  } catch (error) {
    console.error("Chat sessions error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
