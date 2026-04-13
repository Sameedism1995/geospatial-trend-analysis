import { cn } from '@/lib/utils';

type PartnerSectionBackgroundProps = {
  className?: string;
};

export function PartnerSectionBackground({ className }: PartnerSectionBackgroundProps) {
  return (
    <div
      className={cn(
        'pointer-events-none absolute inset-0 z-0 min-h-full w-full overflow-hidden bg-[#fdf7f4]',
        className,
      )}
      aria-hidden="true"
    >
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: 'radial-gradient(circle at center, #FFF991 0%, transparent 70%)',
          opacity: 0.6,
          mixBlendMode: 'multiply',
        }}
      />
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `
            repeating-linear-gradient(0deg, transparent, transparent 5px, rgba(75, 85, 99, 0.06) 5px, rgba(75, 85, 99, 0.06) 6px, transparent 6px, transparent 15px),
            repeating-linear-gradient(90deg, transparent, transparent 5px, rgba(75, 85, 99, 0.06) 5px, rgba(75, 85, 99, 0.06) 6px, transparent 6px, transparent 15px),
            repeating-linear-gradient(0deg, transparent, transparent 10px, rgba(107, 114, 128, 0.04) 10px, rgba(107, 114, 128, 0.04) 11px, transparent 11px, transparent 30px),
            repeating-linear-gradient(90deg, transparent, transparent 10px, rgba(107, 114, 128, 0.04) 10px, rgba(107, 114, 128, 0.04) 11px, transparent 11px, transparent 30px)
          `,
        }}
      />
    </div>
  );
}

export const Component = PartnerSectionBackground;

export default PartnerSectionBackground;
