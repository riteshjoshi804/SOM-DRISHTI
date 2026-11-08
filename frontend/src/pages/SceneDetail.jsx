import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { NA, deg, int, num, pct, withUnit } from '../format.js'
import Histogram from '../components/Histogram.jsx'
import { Button, Card, ErrorNote, ImagePanel, KeyValue, Loading, SectionTitle } from '../components/ui.jsx'
import { RunOptions, useRunOptions } from '../components/RunOptions.jsx'
import { useRunScene } from './Scenes.jsx'

function crsName(wkt) {
  if (!wkt) return 'None (not georeferenced)'
  const m = wkt.match(/^\w+\["([^"]+)"/)
  return m ? m[1] : 'present'
}

function RasterBlock({ label, info, stats }) {
  return (
    <Card>
      <h3 className="mb-3 font-semibold">{label}</h3>
      <KeyValue
        rows={[
          ['Size', `${info.width} × ${info.height} px`],
          ['Bands / dtype', `${info.band_count} / ${info.dtype}`],
          ['NoData', info.nodata ?? 'None declared'],
          ['CRS', crsName(info.crs) + (info.crs_label_consistent === false ? ' (label inconsistent with geotransform)' : '')],
          ['Pixel size', info.pixel_size ? info.pixel_size.map((v) => +v.toPrecision(6)).join(' × ') : 'None'],
          ['Embedded GCPs', info.gcps.length],
          ['Valid pixels', pct(stats.valid_fraction, 2)],
          ['NaN / Inf / zero', `${int(stats.nan_count)} / ${int(stats.inf_count)} / ${int(stats.zero_count)}`],
          ['Min / max', `${num(stats.minimum, 3)} / ${num(stats.maximum, 3)}`],
          ['Mean ± std', `${num(stats.mean, 3)} ± ${num(stats.std, 3)}`],
          ['p1 / p50 / p99', `${num(stats.percentiles.p1, 3)} / ${num(stats.percentiles.p50, 3)} / ${num(stats.percentiles.p99, 3)}`],
        ]}
      />
      <div className="mt-4">
        <Histogram title="Valid-pixel value distribution" counts={stats.histogram_counts} edges={stats.histogram_edges} />
      </div>
    </Card>
  )
}

function CornerTable({ meta }) {
  const names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
  if (!meta.corners) return <p className="text-sm text-muted">No corner coordinates in the XML.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink-2">
          <tr>
            <th className="py-1 pr-4 font-medium">XML corner</th>
            <th className="py-1 pr-4 text-right font-medium">Latitude</th>
            <th className="py-1 pr-4 text-right font-medium">Longitude</th>
            <th className="py-1 pr-4 text-right font-medium">Easting (m)</th>
            <th className="py-1 text-right font-medium">Northing (m)</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {names.map((n) => (
            <tr key={n} className="border-t border-white/5">
              <td className="py-1 pr-4 text-ink-2">{n.replace('_', ' ')}</td>
              <td className="py-1 pr-4 text-right">{num(meta.corners[n]?.latitude, 6)}</td>
              <td className="py-1 pr-4 text-right">{num(meta.corners[n]?.longitude, 6)}</td>
              <td className="py-1 pr-4 text-right">{num(meta.corners_en?.[n]?.easting, 3)}</td>
              <td className="py-1 text-right">{num(meta.corners_en?.[n]?.northing, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">
        Refined corners: {meta.corners_refined ? 'present' : 'not in XML'} · refined easting/northing:{' '}
        {meta.corners_refined_en ? 'present' : 'not in XML'}. The XML corner names follow the original product
        orientation; the mentor-supplied source TIFFs may be flipped relative to them (see embedded GCPs).
      </p>
    </div>
  )
}

export default function SceneDetail({ sceneId }) {
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState(null)
  const { options, update, matchers } = useRunOptions()
  const { run, busy, error: runError } = useRunScene(options)

  useEffect(() => {
    api.scene(sceneId).then(setDetail).catch(setError)
  }, [sceneId])

  const meta = detail?.metadata
  const fp = detail?.predicted_footprint
  return (
    <div>
      <a href="#/scenes" className="text-sm text-accent hover:underline">
        ← All scenes
      </a>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-2">{detail?.scene.sensor || 'Scene'}</div>
          <h1 className="break-all font-mono text-xl font-semibold">{sceneId}</h1>
        </div>
        <Button onClick={() => run(sceneId)} disabled={Boolean(busy) || !detail}>
          {busy ? 'STARTING…' : 'RUN REGISTRATION'}
        </Button>
      </div>
      <ErrorNote error={error || runError} />
      <Card className="mt-4">
        <RunOptions options={options} update={update} matchers={matchers} />
      </Card>
      {!detail && !error && <Loading what="Loading rasters and metadata (first load can take a few seconds)" />}
      {detail && (
        <>
          <SectionTitle hint="Percentile-stretched (1–99%) previews; the original data are untouched">Preview images</SectionTitle>
          <div className="grid gap-4 md:grid-cols-[1fr_2fr]">
            <ImagePanel title="Source" src={api.previewUrl(sceneId, 'source')} caption={detail.scene.source_path.split(/[\\/]/).pop()} />
            <ImagePanel
              title={`Reference (${detail.scene.reference_sensor || 'reference'})`}
              src={api.previewUrl(sceneId, 'reference')}
              caption={detail.scene.reference_path.split(/[\\/]/).pop()}
            />
          </div>

          <SectionTitle>Raster statistics</SectionTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            <RasterBlock label="Source" info={detail.source_info} stats={detail.source_stats} />
            <RasterBlock label="Reference" info={detail.reference_info} stats={detail.reference_stats} />
          </div>

          <SectionTitle hint="Only values present in the XML are shown">XML metadata</SectionTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <KeyValue
                rows={[
                  ['Sensor', `${meta.sensor ?? NA} (inferred from ${meta.sensor_basis ?? NA})`],
                  ['Resolution_in_meter', withUnit(meta.resolution_m, 'm')],
                  ['Sun azimuth', deg(meta.sun_azimuth_deg, 4)],
                  ['Sun elevation', deg(meta.sun_elevation_deg, 4)],
                  ['Solar incidence', deg(meta.solar_incidence_angle_deg, 4)],
                  ['Roll / pitch / yaw', `${deg(meta.roll_deg, 3)} / ${deg(meta.pitch_deg, 3)} / ${deg(meta.yaw_deg, 3)}`],
                  ['Spacecraft altitude', withUnit(meta.spacecraft_altitude_km, 'km')],
                  ['Projection / area', `${meta.projection ?? NA} / ${meta.area ?? NA}`],
                  ['Product size', `${meta.product_image_width ?? NA} × ${meta.product_image_height ?? NA}`],
                  ['Data type', meta.data_type ?? NA],
                  ['Date of pass', meta.date_of_pass ?? NA],
                  ['Imaging orbit', meta.imaging_orbit_number ?? NA],
                  ['Yaw direction flag', meta.spacecraft_yaw_direction ?? NA],
                  ['Reference used', meta.reference_used ?? NA],
                  ['Duplicated tags', `${meta.duplicated_tag_count} (conflicting: ${meta.duplicate_conflicts.length || 'none'})`],
                ]}
              />
            </Card>
            <Card>
              <CornerTable meta={meta} />
            </Card>
          </div>

          <SectionTitle hint="Plausibility reference only - not ground truth">Metadata-predicted footprint</SectionTitle>
          <Card>
            {fp ? (
              <KeyValue
                rows={[
                  ['Corner basis', fp.corner_basis],
                  ['Map coordinates', fp.map_coordinate_basis],
                  ['Predicted scale x / y', `${num(fp.scale_x, 3)} / ${num(fp.scale_y, 3)}`],
                  ['Predicted rotation', deg(fp.rotation_deg, 1)],
                  ['Mirrored (per metadata)', fp.mirrored ? 'yes' : 'no'],
                  ['Corners inside reference', `${fp.corners_inside_reference} / 4`],
                  ...fp.notes.map((n, i) => [`Note ${i + 1}`, n]),
                ]}
              />
            ) : (
              <p className="text-sm text-muted">Cannot be predicted: the reference has no geotransform or the XML has no corners.</p>
            )}
          </Card>

          {detail.overlap_estimate && (
            <>
              <SectionTitle hint="Where a learned matcher will look, with a safety margin; a prior, not a result">Estimated overlap and search region</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Search', detail.overlap_estimate.mode === 'metadata' ? 'around the metadata footprint' : 'the whole reference'],
                    ['Why', detail.overlap_estimate.reason],
                    ['Reference body (from its CRS)', detail.overlap_estimate.coordinate_system?.body ?? NA],
                    [
                      'CRS kind / projection / units',
                      detail.overlap_estimate.coordinate_system
                        ? `${detail.overlap_estimate.coordinate_system.crs_kind} / ${detail.overlap_estimate.coordinate_system.projection ?? NA} / ${detail.overlap_estimate.coordinate_system.units ?? NA}`
                        : NA,
                    ],
                    ['Search radius', withUnit(detail.overlap_estimate.search_radius_px, 'px', 0)],
                    ['Search region on the reference (x0, y0, x1, y1)', detail.overlap_estimate.search_bbox_px?.join(', ') ?? NA],
                    ['Prior footprint inside the reference', pct(detail.overlap_estimate.footprint_fraction_in_reference)],
                    ['Prior footprint on valid pixels', pct(detail.overlap_estimate.footprint_fraction_on_valid_reference)],
                    ...detail.overlap_estimate.warnings.map((w, i) => [`Note ${i + 1}`, w]),
                  ]}
                />
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}
