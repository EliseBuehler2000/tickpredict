import SwiftUI

struct GuideView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Card {
                    Text("Remove it safely").font(.rounded(.title3))
                    VStack(alignment: .leading, spacing: 6) {
                        NumberedLine(1, "Grip the tick with fine-tipped tweezers, as close to the skin as you can.")
                        NumberedLine(2, "Pull straight up with steady pressure. Don't twist or jerk.")
                        NumberedLine(3, "Clean the bite and your hands with soap or rubbing alcohol.")
                        NumberedLine(4, "Keep the tick for scanning. Never crush it with your fingers.")
                    }
                    Text("Skip nail polish, petroleum jelly, or heat. Get it out quickly instead.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.muted)
                }

                Card {
                    Text("Nymph or adult?").font(.rounded(.title3))
                    HStack(spacing: 8) {
                        SizeTile(name: "Nymph", detail: "Poppy seed size") {
                            Circle().fill(Color(light: 0x3B2A20, dark: 0xC9B6A6)).frame(width: 8, height: 8)
                        }
                        SizeTile(name: "Adult female", detail: "Sesame seed size, unfed") {
                            Ellipse().fill(Color(light: 0x8E3B26, dark: 0xD9876B)).frame(width: 16, height: 22)
                                .overlay(alignment: .top) {
                                    Ellipse().fill(Color(light: 0x2B1E16, dark: 0x8A7466)).frame(width: 12, height: 9)
                                }
                        }
                        SizeTile(name: "Adult male", detail: "Sesame seed size, stays flat") {
                            Ellipse().fill(Color(light: 0x2B1E16, dark: 0x8A7466)).frame(width: 14, height: 19)
                        }
                    }
                    Text("Nymphs are tiny and easy to miss, so they often stay attached longer. In deer ticks, they cause most Lyme infections in people. Males rarely feed long.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.body)
                }

                Card {
                    Text("What engorgement means").font(.rounded(.title3))
                    HStack(alignment: .bottom, spacing: 8) {
                        ForEach(Array(zip(Engorgement.order.indices, Engorgement.order)), id: \.0) { index, label in
                            VStack(spacing: 6) {
                                Ellipse()
                                    .fill(Theme.engorgementRamp[index])
                                    .frame(width: CGFloat(18 + index * 12), height: CGFloat(24 + index * 12))
                                Text(Engorgement(label).shortName == "None" ? "Unengorged" : Engorgement(label).shortName)
                                    .font(.footnote.weight(.semibold))
                                Text(["Flat, little or no blood", "Rounder, partly fed", "Swollen, fed for days"][index])
                                    .font(.caption)
                                    .foregroundStyle(Theme.muted)
                                    .multilineTextAlignment(.center)
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                    Text("The longer a tick feeds, the more it swells, and the more chance it has to pass on an infection. Tell your doctor if yours looked engorged.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.body)
                }

                Card(fill: Theme.accentTint) {
                    Text("When to call a doctor").font(.rounded(.title3))
                    VStack(alignment: .leading, spacing: 6) {
                        Bullet("A rash, especially one that spreads or looks like a bull's-eye")
                        Bullet("Fever, chills, headache, tiredness, or achy joints within 30 days")
                        Bullet("You couldn't remove the whole tick")
                        Bullet("The tick was engorged or attached for a long time")
                    }
                    NavigationLink(value: Route.findCare) {
                        Label("Find care nearby", systemImage: "mappin.and.ellipse")
                            .font(.headline)
                            .frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.glassProminent)
                }

                Card {
                    Text("About TickPredict").font(.rounded(.title3))
                    Text("Two image models, trained on about 13,000 iNaturalist phone photos and Penn research photos. They know 8 common U.S. ticks: blacklegged (deer), western blacklegged, American dog, lone star, Gulf Coast, brown dog, Rocky Mountain wood, and Asian longhorned.")
                    Text("On photos held out from training, they told blacklegged ticks from others 97% of the time, named the exact species 85%, life stage 93–96%, and engorgement 86%.")
                    Text("They can't tell larvae or other kinds of tick, and confidence is the model's own score, not a guarantee.")
                    Text("Everything runs on your iPhone. Photos never leave it.")
                    Text("Models v2.0 · engorgement v1.0").font(.footnote).foregroundStyle(Theme.muted)
                }
                .font(.subheadline)
                .foregroundStyle(Theme.body)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
        .background(Theme.ground)
        .navigationTitle("Tick guide")
    }
}

private struct SizeTile<Mark: View>: View {
    let name: String
    let detail: String
    @ViewBuilder var mark: Mark

    var body: some View {
        VStack(spacing: 8) {
            mark.frame(height: 40)
            Text(name).font(.subheadline.weight(.semibold))
            Text(detail).font(.footnote).foregroundStyle(Theme.muted).multilineTextAlignment(.center)
        }
        .padding(14)
        .frame(maxWidth: .infinity)
        .background(Theme.ground, in: .rect(cornerRadius: 14))
    }
}

private struct Bullet: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("•").foregroundStyle(Theme.accent)
            Text(text)
        }
        .font(.subheadline)
        .foregroundStyle(Theme.body)
    }
}
