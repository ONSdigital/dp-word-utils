package com.github.onsdigital.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/**
 * @author sullid (David Sullivan) on 20/02/2018
 * @project babbage
 */
public class GZipFile {

    private final String filename;

    public GZipFile(String filename) {
        this.filename = filename;
    }

    public File gunzip() throws IOException {
        String outputName = this.filename.replaceAll(".gz", "");
        return this.gunzip(outputName);
    }

    public File gunzip(String outputFile) throws IOException {

        byte[] buffer = new byte[1024];

        // Auto closable
        try (GZIPInputStream gzis =
                     new GZIPInputStream(new FileInputStream(filename))) {

            // Auto closable
            try (FileOutputStream out =
                         new FileOutputStream(outputFile)) {

                int len;
                while ((len = gzis.read(buffer)) > 0) {
                    out.write(buffer, 0, len);
                }
            }
        }

        return new File(outputFile);
    }

}
